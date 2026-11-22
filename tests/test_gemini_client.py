import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_client import (  # noqa: E402
    DEFAULT_MODEL,
    FATAL_KINDS,
    GeminiClient,
    GeminiError,
    GeminiSettings,
    classify_error,
    load_settings,
    resolve_model,
)

KEY = "SECRET-KEY-12345"


@pytest.fixture
def settings():
    return GeminiSettings(api_key=KEY, model="test-model", timeout_s=30)


# ---- settings -------------------------------------------------------------


def test_env_key_wins_over_secrets():
    s = load_settings(secrets={"GEMINI_API_KEY": "from-secrets"}, env={"GEMINI_API_KEY": "from-env"})
    assert s.api_key == "from-env"


def test_falls_back_to_streamlit_secrets():
    s = load_settings(secrets={"GEMINI_API_KEY": "from-secrets"}, env={})
    assert s.api_key == "from-secrets"


def test_google_api_key_env_is_accepted():
    assert load_settings(env={"GOOGLE_API_KEY": "g-key"}).api_key == "g-key"


@pytest.mark.parametrize("secrets", [None, {}, {"GEMINI_API_KEY": "   "}, {"OTHER": "x"}])
def test_missing_key_gives_a_clear_error(secrets):
    with pytest.raises(GeminiError) as exc:
        load_settings(secrets=secrets, env={})
    assert exc.value.kind == "missing_key" and "GEMINI_API_KEY" in exc.value.message
    assert ".streamlit/secrets.toml" in exc.value.message and "missing_key" in FATAL_KINDS


def test_model_is_configurable_and_has_a_default():
    assert resolve_model(env={}) == DEFAULT_MODEL
    assert resolve_model(secrets={"SMARTGRADER_MODEL": "from-secrets"}, env={}) == "from-secrets"
    assert resolve_model(secrets={"SMARTGRADER_MODEL": "from-secrets"}, env={"SMARTGRADER_MODEL": "env-model"}) == "env-model"
    assert load_settings(env={"GEMINI_API_KEY": "k", "SMARTGRADER_MODEL": "m"}).model == "m"


def test_timeout_setting_is_parsed_and_clamped():
    base = {"GEMINI_API_KEY": "k"}
    assert load_settings(env=base).timeout_s == 30
    assert load_settings(env={**base, "SMARTGRADER_TIMEOUT_SECONDS": "45"}).timeout_s == 45
    assert load_settings(env={**base, "SMARTGRADER_TIMEOUT_SECONDS": "1"}).timeout_s == 5
    assert load_settings(env={**base, "SMARTGRADER_TIMEOUT_SECONDS": "9999"}).timeout_s == 120
    assert load_settings(env={**base, "SMARTGRADER_TIMEOUT_SECONDS": "abc"}).timeout_s == 30


def test_key_is_never_in_repr(settings):
    assert KEY not in repr(settings) and KEY not in str(settings)


# ---- error classification -------------------------------------------------


class FakeAPIError(Exception):
    def __init__(self, code, message="boom"):
        super().__init__(message)
        self.code = code


class ReadTimeout(Exception):  # name mimics httpx.ReadTimeout
    pass


class ConnectError(Exception):  # name mimics httpx.ConnectError
    pass


@pytest.mark.parametrize(
    "exc, kind, retryable",
    [
        (FakeAPIError(401), "auth", False),
        (FakeAPIError(403), "auth", False),
        (FakeAPIError(400, "API key not valid. Please pass a valid API key."), "auth", False),
        (FakeAPIError(404), "model_not_found", False),
        (FakeAPIError(429), "rate_limit", True),
        (FakeAPIError(500), "server", True),
        (FakeAPIError(503), "server", True),
        (FakeAPIError(504), "timeout", True),
        (FakeAPIError(400, "bad schema"), "bad_request", False),
        (ReadTimeout("read timed out"), "timeout", True),
        (ConnectError("dns failure"), "network", True),
        (RuntimeError("weird"), "unknown", True),
    ],
)
def test_classification(settings, exc, kind, retryable):
    err = classify_error(exc, settings)
    assert err.kind == kind and err.retryable is retryable


def test_messages_are_clear(settings):
    assert "30 seconds" in classify_error(ReadTimeout("x"), settings).message
    assert "test-model" in classify_error(FakeAPIError(404), settings).message
    assert "SMARTGRADER_MODEL" in classify_error(FakeAPIError(404), settings).message
    assert "internet" in classify_error(ConnectError("x"), settings).message
    assert "rate-limiting" in classify_error(FakeAPIError(429), settings).message


def test_api_key_is_scrubbed_from_error_messages(settings):
    for exc in [
        FakeAPIError(400, f"bad request ?key={KEY}"),
        RuntimeError(f"connection to https://x/?key={KEY} failed"),
    ]:
        assert KEY not in classify_error(exc, settings).message


def test_existing_gemini_error_passes_through(settings):
    original = GeminiError("auth", "msg")
    assert classify_error(original, settings) is original


# ---- the client with a fake SDK ------------------------------------------


class FakeSDK:
    """Mimics client.models.generate_content; records every call."""

    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        b = self.behaviours.pop(0) if len(self.behaviours) > 1 else self.behaviours[0]
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(text=b)


def make_client(settings, *behaviours):
    sdk = FakeSDK(behaviours)
    return GeminiClient(settings, sdk), sdk


def test_generate_json_sends_the_right_request(settings):
    client, sdk = make_client(settings, '{"ok": true}')
    out = client.generate_json("SYSTEM", "USER", {"type": "object"})
    assert out == '{"ok": true}'
    call = sdk.calls[0]
    assert call["model"] == "test-model" and call["contents"] == "USER"
    cfg = call["config"]
    assert cfg["system_instruction"] == "SYSTEM"
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_json_schema"] == {"type": "object"}
    assert cfg["temperature"] == 0.0 and cfg["max_output_tokens"] > 0
    assert cfg["thinking_config"] == {"thinking_budget": 0}  # fastest setting first


def test_empty_or_blocked_response(settings):
    for text in (None, "", "   "):
        client, _ = make_client(settings, text)
        with pytest.raises(GeminiError) as exc:
            client.generate_json("s", "u", {})
        assert exc.value.kind == "empty_response" and exc.value.retryable


def test_api_errors_become_gemini_errors_without_the_key(settings):
    client, _ = make_client(settings, FakeAPIError(401, f"invalid key {KEY}"))
    with pytest.raises(GeminiError) as exc:
        client.generate_json("s", "u", {})
    assert exc.value.kind == "auth" and KEY not in exc.value.message
    assert exc.value.__cause__ is None  # original exception text is not chained into tracebacks


def test_timeout_becomes_a_clear_error(settings):
    client, _ = make_client(settings, ReadTimeout("The read operation timed out"))
    with pytest.raises(GeminiError) as exc:
        client.generate_json("s", "u", {})
    assert exc.value.kind == "timeout" and "30 seconds" in exc.value.message


def test_thinking_setting_falls_back_when_the_model_rejects_it(settings):
    rejected = FakeAPIError(400, "Unsupported thinking_budget for this model")
    client, sdk = make_client(settings, rejected, '{"ok": 1}')
    assert client.generate_json("s", "u", {}) == '{"ok": 1}'
    assert sdk.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}
    assert sdk.calls[1]["config"]["thinking_config"] == {"thinking_level": "low"}
    # the working choice is remembered, so later requests don't retry the rejected one
    client.generate_json("s", "u", {})
    assert sdk.calls[2]["config"]["thinking_config"] == {"thinking_level": "low"}


def test_thinking_fallback_ends_without_thinking_config_then_gives_up(settings):
    rejected = FakeAPIError(400, "thinking is not supported")
    client, sdk = make_client(settings, rejected)  # rejects every time
    with pytest.raises(GeminiError) as exc:
        client.generate_json("s", "u", {})
    assert exc.value.kind == "bad_request"
    assert len(sdk.calls) == 3 and "thinking_config" not in sdk.calls[2]["config"]


def test_create_reports_a_missing_sdk(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)  # makes `from google import genai` fail
    with pytest.raises(GeminiError) as exc:
        GeminiClient.create(settings)
    assert exc.value.kind == "sdk_missing" and "pip install" in exc.value.message


# ---- key source and quote stripping ---------------------------------------


def test_key_source_is_reported():
    env_first = load_settings(secrets={"GEMINI_API_KEY": "s"}, env={"GEMINI_API_KEY": "e"})
    assert "environment variable" in env_first.key_source
    from_file = load_settings(secrets={"GEMINI_API_KEY": "s"}, env={})
    assert "secrets.toml" in from_file.key_source
    assert "GOOGLE_API_KEY" in load_settings(env={"GOOGLE_API_KEY": "g"}).key_source


@pytest.mark.parametrize("pasted", ['"AIzaKEY"', "'AIzaKEY'", '  "AIzaKEY"  ', "AIzaKEY "])
def test_quotes_and_spaces_around_the_key_are_removed(pasted):
    assert load_settings(env={"GEMINI_API_KEY": pasted}).api_key == "AIzaKEY"
    assert load_settings(secrets={"GEMINI_API_KEY": pasted}, env={}).api_key == "AIzaKEY"


def test_auth_error_names_the_key_source_but_not_the_key():
    s = GeminiSettings(api_key=KEY, key_source="the GEMINI_API_KEY environment variable")
    msg = classify_error(FakeAPIError(401), s).message
    assert "the GEMINI_API_KEY environment variable" in msg and KEY not in msg
