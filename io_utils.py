"""Splitting pasted answers and building the CSV export."""

from __future__ import annotations

import csv
import io
import re

from models import GradedAnswer, Rubric

# A separator is a line containing only three or more dashes (e.g. '---').
_SEPARATOR = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)

MAX_ANSWERS = 100  # keeps a prototype run cheap and predictable


def split_answers(text: str, max_answers: int = MAX_ANSWERS) -> list[str]:
    """Split pasted text into individual answers on '---' lines.

    Blank answers are dropped. Raises ValueError if there are too many.
    """
    parts = [p.strip() for p in _SEPARATOR.split(text or "")]
    answers = [p for p in parts if p]
    if len(answers) > max_answers:
        raise ValueError(
            f"{len(answers)} answers pasted; the prototype limit is {max_answers}."
        )
    return answers


def escape_cell(value: object) -> str:
    """Neutralise spreadsheet formula injection.

    A student answer starting with =, +, -, @ (or a tab/CR) could be executed
    as a formula when the CSV is opened in Excel/Sheets. Prefixing a single
    quote makes spreadsheets treat it as text.
    """
    s = "" if value is None else str(value)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


def fmt_number(x: float | None) -> str:
    """Render 4.0 as '4' and 3.5 as '3.5'; None as ''."""
    if x is None:
        return ""
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def build_csv(rubric: Rubric, rows: list[GradedAnswer]) -> str:
    """Return the export as a CSV string (UTF-8; add a BOM at download time)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "student",
            "answer",
            "ai_score",
            "final_score",
            "max_score",
            "reviewed",
            "needs_human_review",
            "review_reason",
            "feedback",
            "ai_breakdown",
            "needs_manual_grading",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                escape_cell(r.label),
                escape_cell(r.answer),
                fmt_number(r.ai_score),
                fmt_number(r.final_score),
                fmt_number(rubric.max_total),
                "yes" if r.reviewed else "no",
                "yes" if (r.needs_review or r.needs_manual) else "no",
                escape_cell(r.review_reason or r.error),
                escape_cell(r.ai_feedback),
                escape_cell(r.ai_breakdown),
                "yes" if r.needs_manual else "no",
            ]
        )
    return buf.getvalue()
