"""Pure helpers that convert between UI tables (pandas) and models.

No Streamlit imports here, so everything is unit-testable.
"""

from __future__ import annotations

import json

import pandas as pd

from io_utils import fmt_number
from models import Criterion, GradedAnswer, Rubric

DEFAULT_RUBRIC_ROWS = [
    {"Criterion": "Accuracy", "Description": "Key facts are correct", "Max points": 3.0},
    {"Criterion": "Clarity", "Description": "Explained in clear words", "Max points": 2.0},
]


def _text(v: object) -> str:
    return "" if v is None or pd.isna(v) else str(v).strip()


def _num(v: object) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _bool(v: object) -> bool:
    return False if v is None or pd.isna(v) else bool(v)


def rubric_from_table(question: str, table: pd.DataFrame) -> tuple[Rubric | None, list[str]]:
    """Build a validated Rubric from the question and the rubric editor table.

    Returns (rubric, errors). `rubric` is None whenever `errors` is non-empty.
    Fully blank rows are ignored.
    """
    errors: list[str] = []
    if not question.strip():
        errors.append("Enter the question.")

    criteria: list[Criterion] = []
    row_problem = False
    for pos, (_, row) in enumerate(table.iterrows(), start=1):
        name = _text(row.get("Criterion"))
        desc = _text(row.get("Description"))
        pts = _num(row.get("Max points"))
        if not name and not desc and pts is None:
            continue
        if not name:
            errors.append(f"Rubric row {pos}: add a criterion name.")
            row_problem = True
        elif pts is None or pts <= 0:
            errors.append(f"Rubric row {pos} ({name}): max points must be greater than 0.")
            row_problem = True
        else:
            criteria.append(Criterion(name=name, description=desc, max_points=pts))

    if not criteria and not row_problem:
        errors.append("Add at least one rubric criterion with points.")

    names = [c.name.lower() for c in criteria]
    if len(names) != len(set(names)):
        errors.append("Rubric criterion names must be unique.")

    if errors:
        return None, errors
    return Rubric(question=question.strip(), criteria=criteria), []


def flag_text(g: GradedAnswer) -> str:
    """The short human-review flag shown in the results table."""
    if g.needs_manual:
        # Surface the real cause in the table. The full message remains in the
        # Answer details panel; this short version helps diagnose API setup,
        # quota and response problems without opening every failed row.
        message = (g.error or "AI grading failed; open Answer details.").removesuffix(
            " Grade this one manually."
        )
        return f"⚠ {message[:160]}{'…' if len(message) > 160 else ''}"
    if g.needs_review:
        return "⚠ " + (g.review_reason or "Please check this answer")
    return ""


def results_to_table(graded: list[GradedAnswer]) -> pd.DataFrame:
    """Turn AI results into the editable review table.

    'Final score' starts as a copy of the AI score so the teacher only edits
    what they disagree with; 'Reviewed' always starts False.
    """
    ai = pd.Series([g.ai_score for g in graded], dtype="float64")
    return pd.DataFrame(
        {
            "Student": [g.label for g in graded],
            "Answer": [g.answer for g in graded],
            "AI score": ai,
            "Final score": ai.copy(),
            "Reviewed": [False] * len(graded),
            "Flag": [flag_text(g) for g in graded],
            "Feedback": [g.ai_feedback for g in graded],
            "AI breakdown": [g.ai_breakdown for g in graded],
            "Needs manual": [g.needs_manual for g in graded],
            "Error": [g.error for g in graded],
            "Needs review": [g.needs_review for g in graded],
            "Review reason": [g.review_reason for g in graded],
            "Breakdown detail": [g.breakdown_json for g in graded],
        }
    )


# Columns the editor doesn't display but the app still needs.
HIDDEN_COLUMNS = ["Needs manual", "Error", "Needs review", "Review reason", "Breakdown detail"]


def restore_hidden_columns(edited: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Re-attach hidden columns if the editor didn't return them.

    The table has a fixed number of rows (no adding or deleting), so the
    originals line up by position.
    """
    out = edited.copy()
    if len(out) == len(original):
        for col in HIDDEN_COLUMNS:
            if col not in out.columns and col in original.columns:
                out[col] = original[col].to_numpy()
    return out


def breakdown_frame(json_text: str) -> pd.DataFrame | None:
    """Per-criterion breakdown as a small table, or None if there isn't one."""
    if not json_text:
        return None
    try:
        rows = json.loads(json_text)
        frame = pd.DataFrame(
            [
                {
                    "Criterion": r["criterion"],
                    "Score": f"{fmt_number(r['score'])} / {fmt_number(r['max'])}",
                    "Reason": r.get("reason", ""),
                }
                for r in rows
            ]
        )
    except (ValueError, KeyError, TypeError):
        return None
    return None if frame.empty else frame


def table_to_graded(table: pd.DataFrame, rubric: Rubric) -> list[GradedAnswer]:
    """Read the (possibly teacher-edited) review table back into models.

    Final scores are clamped to [0, rubric.max_total] so a typo can't export
    an impossible grade.
    """
    out: list[GradedAnswer] = []
    for i, (_, r) in enumerate(table.iterrows(), start=1):
        final = _num(r.get("Final score"))
        if final is not None:
            final = min(max(final, 0.0), rubric.max_total)
        out.append(
            GradedAnswer(
                label=_text(r.get("Student")) or f"Student {i}",
                answer=_text(r.get("Answer")),
                ai_score=_num(r.get("AI score")),
                ai_feedback=_text(r.get("Feedback")),
                ai_breakdown=_text(r.get("AI breakdown")),
                needs_manual=_bool(r.get("Needs manual")),
                error=_text(r.get("Error")),
                needs_review=_bool(r.get("Needs review")),
                review_reason=_text(r.get("Review reason")),
                breakdown_json=_text(r.get("Breakdown detail")),
                final_score=final,
                reviewed=_bool(r.get("Reviewed")),
            )
        )
    return out
