"""Data models for SmartGrader.

Design rule: the AI only *suggests*. A teacher's final score is always a
separate field that the teacher controls.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


class Criterion(BaseModel):
    """One row of the rubric."""

    name: str = Field(min_length=1)
    description: str = ""
    max_points: float = Field(gt=0)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Criterion name cannot be empty")
        return v


class Rubric(BaseModel):
    question: str = Field(min_length=1)
    criteria: list[Criterion] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _unique_names(cls, v: list[Criterion]) -> list[Criterion]:
        names = [c.name.lower() for c in v]
        if len(names) != len(set(names)):
            raise ValueError("Criterion names must be unique")
        return v

    @property
    def max_total(self) -> float:
        return sum(c.max_points for c in self.criteria)


class CriterionScore(BaseModel):
    """What Gemini returns for a single criterion."""

    criterion: str
    score: float  # range is enforced by clamp_suggestion, not at parse time
    reason: str = ""


class Suggestion(BaseModel):
    """The full structured response expected from Gemini for one answer."""

    criterion_scores: list[CriterionScore]
    feedback: str = ""
    needs_human_review: bool = False  # the AI asking the teacher to look closely
    review_reason: str = ""


class GradedAnswer(BaseModel):
    """One student answer plus the AI suggestion and the teacher's decision."""

    label: str
    answer: str
    ai_score: float | None = None
    ai_feedback: str = ""
    ai_breakdown: str = ""  # short human-readable per-criterion summary
    breakdown_json: str = ""  # per-criterion score + reason, for the details view
    needs_review: bool = False  # AI (or a safety check) wants a human to look closely
    review_reason: str = ""
    needs_manual: bool = False  # True when the AI call failed or was invalid
    error: str = ""
    error_kind: str = ""  # e.g. "timeout", "auth"; empty when grading worked
    final_score: float | None = None
    reviewed: bool = False


def clamp_suggestion(rubric: Rubric, suggestion: Suggestion) -> tuple[Suggestion, float]:
    """Validate a suggestion against the rubric and return (clean, total).

    - Every rubric criterion must be present exactly once (matched by name,
      case-insensitively); unknown criteria are rejected.
    - Scores are clamped to ``[0, max_points]``.
    Raises ValueError if the model's output doesn't line up with the rubric,
    so the caller can retry or fall back to manual grading.
    """
    by_name = {c.name.lower(): c for c in rubric.criteria}
    seen: dict[str, CriterionScore] = {}

    for cs in suggestion.criterion_scores:
        key = cs.criterion.strip().lower()
        if key not in by_name:
            raise ValueError(f"Unknown criterion in model output: {cs.criterion!r}")
        if key in seen:
            raise ValueError(f"Duplicate criterion in model output: {cs.criterion!r}")
        seen[key] = cs

    missing = [c.name for c in rubric.criteria if c.name.lower() not in seen]
    if missing:
        raise ValueError(f"Model output missing criteria: {', '.join(missing)}")

    cleaned: list[CriterionScore] = []
    total = 0.0
    for c in rubric.criteria:  # keep rubric order
        cs = seen[c.name.lower()]
        score = min(max(cs.score, 0.0), c.max_points)
        total += score
        cleaned.append(CriterionScore(criterion=c.name, score=score, reason=cs.reason.strip()))

    clean = Suggestion(
        criterion_scores=cleaned,
        feedback=suggestion.feedback.strip(),
        needs_human_review=suggestion.needs_human_review,
        review_reason=suggestion.review_reason.strip(),
    )
    return clean, total


def format_breakdown(rubric: Rubric, suggestion: Suggestion) -> str:
    """e.g. 'Accuracy 2/3; Clarity 1/2'."""
    maxes = {c.name: c.max_points for c in rubric.criteria}
    return "; ".join(
        f"{cs.criterion} {_fmt(cs.score)}/{_fmt(maxes[cs.criterion])}"
        for cs in suggestion.criterion_scores
    )


def format_breakdown_json(rubric: Rubric, suggestion: Suggestion) -> str:
    """Per-criterion rows (score, max, reason) as JSON, for the details view."""
    maxes = {c.name: c.max_points for c in rubric.criteria}
    return json.dumps(
        [
            {
                "criterion": cs.criterion,
                "score": cs.score,
                "max": maxes[cs.criterion],
                "reason": cs.reason,
            }
            for cs in suggestion.criterion_scores
        ]
    )


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"
