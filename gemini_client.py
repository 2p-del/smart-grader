"""Everything that talks to the Gemini API, in one reusable module.

It knows nothing about rubrics or grading: give it a system prompt, a user
prompt and a JSON schema and it returns the raw JSON text, or raises a
`GeminiError` whose message is safe and clear enough to show a teacher.

Settings (key, model, timeout) come from environment variables or Streamlit
secrets. The key is held in memory on the server, is excluded from every
repr, and is scrubbed from every error message.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_S = 30.0
MIN_TIMEOUT_S, MAX_TIMEOUT_S = 5.0, 120.0
DEFAULT_MAX_OUTPUT_TOKENS = 1500

# Problems that affect the whole run, so grading stops instead of sending more
# requests that are expected to fail (including provider quota/rate limits).
FATAL_KINDS = frozenset({"missing_key", "sdk_missing", "auth", "model_not_found", "rate_limit"})


class GeminiError(Exception):
    """A Gemini failure with a teacher-readable message."""

    def __init__(self, kind: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str = field(repr=False)  # never printed, even by accident
    key_source: str = "unknown"  # where the key came from, for diagnosing mix-ups (not secret)
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


# ---- settings -------------------------------------------------------------


def _lookup(name: str, env: Mapping[str, str], secrets: Mapping[str, Any] | None) -> str:
    value = (env.get(name) or "").strip()
    if value:
        return value
    if secrets:
        try:
            return str(secrets.get(name) or "").strip()
        except Exception:
            return ""
    return ""


def resolve_model(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> str:
    """The configured model name: SMARTGRADER_MODEL (env, then secrets), else the default."""
    env = os.environ if env is None else env
    return _lookup("SMARTGRADER_MODEL", env, secrets) or DEFAULT_MODEL


def load_settings(
    secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None
) -> GeminiSettings:
    """Read settings from the environment first, then Streamlit secrets.

    Raises GeminiError("missing_key") if no API key is configured.
    """
    env = os.environ if env is None else env

    def clean(value: Any) -> str:
        # Strip spaces and any quote marks pasted along with the key.
        return str(value or "").strip().strip("\"'").strip()

    secret_key = ""
    if secrets:
        try:
            secret_key = clean(secrets.get("GEMINI_API_KEY"))
        except Exception:
            secret_key = ""
    key, source = "", ""
    for candidate, label in (
        (clean(env.get("GEMINI_API_KEY")), "the GEMINI_API_KEY environment variable"),
        (secret_key, "the GEMINI_API_KEY entry in .streamlit/secrets.toml"),
        (clean(env.get("GOOGLE_API_KEY")), "the GOOGLE_API_KEY environment variable"),
    ):
        if candidate:
            key, source = candidate, label
            break
    if not key:
        raise GeminiError(
            "missing_key",
            "No Gemini API key found. Set the GEMINI_API_KEY environment variable or add it to "
            ".streamlit/secrets.toml, then restart the app.",
        )
    try:
        timeout = float(_lookup("SMARTGRADER_TIMEOUT_SECONDS", env, secrets) or DEFAULT_TIMEOUT_S)
    except ValueError:
        timeout = DEFAULT_TIMEOUT_S
    timeout = min(max(timeout, MIN_TIMEOUT_S), MAX_TIMEOUT_S)
    return GeminiSettings(
        api_key=key, key_source=source, model=resolve_model(secrets, env), timeout_s=timeout
    )


# ---- error handling -------------------------------------------------------


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def classify_error(exc: BaseException, settings: GeminiSettings) -> GeminiError:
    """Turn any SDK or network exception into a clear, key-free GeminiError.

    Uses duck typing (an integer `code` attribute, class names) so it works
    with whatever exception classes the SDK raises, and is testable without it.
    """
    if isinstance(exc, GeminiError):
        return exc

    msg = _redact(str(exc), settings.api_key)
    low = msg.lower()
    code = getattr(exc, "code", None)
    code = code if isinstance(code, int) else None
    names = {c.__name__.lower() for c in type(exc).__mro__}

    if code in (408, 504) or any("timeout" in n for n in names):
        return GeminiError(
            "timeout",
            f"Gemini did not answer within {settings.timeout_s:g} seconds.",
            retryable=True,
        )
    if code is not None:
        if code in (401, 403) or (code == 400 and "api key" in low):
            return GeminiError(
                "auth",
                "Gemini rejected the API key (invalid, expired, or not allowed for this model). "
                f"The app is using the key from {settings.key_source}. "
                "Check that key and restart the app.",
            )
        if code == 404:
            return GeminiError(
                "model_not_found",
                f"Gemini could not find the model '{settings.model}'. "
                "Set SMARTGRADER_MODEL to a model your key can use.",
            )
        if code == 429:
            return GeminiError(
                "rate_limit",
                "Gemini's request limit or quota has been reached. SmartGrader stopped this run "
                "to avoid sending more blocked requests. Wait for the limit to reset, or check the "
                "API key's usage and quota in Google AI Studio.",
                retryable=True,
            )
        if code >= 500:
            return GeminiError(
                "server", f"Gemini had a temporary problem (HTTP {code}).", retryable=True
            )
        return GeminiError("bad_request", f"Gemini rejected the request: {msg[:200]}")
    if names & {"connecterror", "networkerror", "connectionerror", "gaierror", "remoteprotocolerror"}:
        return GeminiError(
            "network", "Could not reach Gemini. Check your internet connection.", retryable=True
        )
    return GeminiError(
        "unknown",
        f"Unexpected problem talking to Gemini ({type(exc).__name__}: {msg[:150]}).",
        retryable=True,
    )


# ---- the client -----------------------------------------------------------

# Grading needs no long reasoning, and "thinking" is the biggest source of
# latency. Try the lowest setting each model accepts, falling back if one is
# rejected. The working choice is remembered for the rest of the session.
_THINKING_OPTIONS: tuple[dict | None, ...] = ({"thinking_budget": 0}, {"thinking_level": "low"}, None)


class GeminiClient:
    """Thread-safe wrapper: `generate_json(system, user, schema) -> str`."""

    def __init__(self, settings: GeminiSettings, sdk_client: Any):
        self.settings = settings
        self._client = sdk_client
        self._thinking_idx = 0
        self._lock = threading.Lock()

    @classmethod
    def create(cls, settings: GeminiSettings) -> "GeminiClient":
        """Build the real SDK client, with a hard request timeout."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise GeminiError(
                "sdk_missing",
                "The google-genai package is not installed. Run: pip install -r requirements.txt",
            ) from None
        try:
            sdk = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(timeout=int(settings.timeout_s * 1000)),  # milliseconds
            )
        except Exception as e:
            raise GeminiError(
                "unknown", f"Could not start the Gemini client ({_redact(str(e), settings.api_key)[:150]})."
            ) from None
        return cls(settings, sdk)

    def _config(self, system: str, schema: dict) -> dict:
        config: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        thinking = _THINKING_OPTIONS[self._thinking_idx]
        if thinking:
            config["thinking_config"] = thinking
        return config

    def _try_next_thinking_option(self, failed_idx: int) -> bool:
        with self._lock:
            if self._thinking_idx == failed_idx and failed_idx < len(_THINKING_OPTIONS) - 1:
                self._thinking_idx += 1
            return self._thinking_idx > failed_idx

    def generate_json(self, system: str, user: str, schema: dict) -> str:
        """One request. Returns the model's JSON text; raises GeminiError on failure."""
        while True:
            idx = self._thinking_idx
            try:
                response = self._client.models.generate_content(
                    model=self.settings.model,
                    contents=user,
                    config=self._config(system, schema),
                )
            except Exception as e:
                err = classify_error(e, self.settings)
                if (
                    err.kind == "bad_request"
                    and "thinking" in str(e).lower()
                    and self._try_next_thinking_option(idx)
                ):
                    continue  # this model rejected that thinking setting; use the next
                raise err from None  # `from None` keeps the original (unredacted) text out of tracebacks

            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise GeminiError(
                    "empty_response",
                    "Gemini returned an empty or blocked response.",
                    retryable=True,
                )
            return text
