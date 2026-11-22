import csv
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import build_csv, escape_cell, split_answers  # noqa: E402
from models import (  # noqa: E402
    Criterion,
    CriterionScore,
    GradedAnswer,
    Rubric,
    Suggestion,
    clamp_suggestion,
    format_breakdown,
)


@pytest.fixture
def rubric():
    return Rubric(
        question="Why is the sky blue?",
        criteria=[
            Criterion(name="Accuracy", description="Mentions scattering", max_points=3),
            Criterion(name="Clarity", description="Clear wording", max_points=2),
        ],
    )


# ---- splitting -----------------------------------------------------------


def test_split_basic():
    assert split_answers("one\n---\ntwo\n---\nthree") == ["one", "two", "three"]


def test_split_drops_blanks_and_whitespace():
    assert split_answers("\n---\n  a  \n----\n\n---\n") == ["a"]


def test_split_keeps_inline_dashes():
    # dashes inside a sentence must not split
    assert split_answers("it is a well--known fact --- yes") == [
        "it is a well--known fact --- yes"
    ]


def test_split_empty():
    assert split_answers("") == []
    assert split_answers("   \n ") == []


def test_split_limit():
    with pytest.raises(ValueError):
        split_answers("\n---\n".join("x" for _ in range(5)), max_answers=3)


# ---- rubric --------------------------------------------------------------


def test_rubric_total(rubric):
    assert rubric.max_total == 5


def test_rubric_rejects_duplicate_names():
    with pytest.raises(ValueError):
        Rubric(
            question="q",
            criteria=[
                Criterion(name="A", max_points=1),
                Criterion(name="a", max_points=1),
            ],
        )


def test_rubric_rejects_zero_points():
    with pytest.raises(ValueError):
        Criterion(name="A", max_points=0)


# ---- clamping / validation ----------------------------------------------


def test_clamp_caps_and_orders(rubric):
    s = Suggestion(
        criterion_scores=[
            CriterionScore(criterion="clarity", score=9, reason="ok"),
            CriterionScore(criterion="Accuracy", score=-1, reason="bad"),
        ],
        feedback=" nice ",
    )
    clean, total = clamp_suggestion(rubric, s)
    assert [c.criterion for c in clean.criterion_scores] == ["Accuracy", "Clarity"]
    assert [c.score for c in clean.criterion_scores] == [0, 2]
    assert total == 2
    assert clean.feedback == "nice"
    assert format_breakdown(rubric, clean) == "Accuracy 0/3; Clarity 2/2"


def test_clamp_rejects_unknown_missing_duplicate(rubric):
    with pytest.raises(ValueError):
        clamp_suggestion(
            rubric,
            Suggestion(criterion_scores=[CriterionScore(criterion="Nope", score=1)]),
        )
    with pytest.raises(ValueError):
        clamp_suggestion(
            rubric,
            Suggestion(criterion_scores=[CriterionScore(criterion="Accuracy", score=1)]),
        )
    with pytest.raises(ValueError):
        clamp_suggestion(
            rubric,
            Suggestion(
                criterion_scores=[
                    CriterionScore(criterion="Accuracy", score=1),
                    CriterionScore(criterion="Accuracy", score=1),
                    CriterionScore(criterion="Clarity", score=1),
                ]
            ),
        )


# ---- CSV -----------------------------------------------------------------


def test_escape_cell():
    assert escape_cell("=1+1") == "'=1+1"
    assert escape_cell("+cmd") == "'+cmd"
    assert escape_cell("-5") == "'-5"
    assert escape_cell("@SUM(A1)") == "'@SUM(A1)"
    assert escape_cell("normal") == "normal"
    assert escape_cell(None) == ""


def test_build_csv_roundtrip_and_escaping(rubric):
    rows = [
        GradedAnswer(
            label="Student 1",
            answer="Rayleigh scattering,\nwith a newline",
            ai_score=4,
            ai_feedback="Good.",
            ai_breakdown="Accuracy 3/3; Clarity 1/2",
            final_score=5,
            reviewed=True,
        ),
        GradedAnswer(label="Student 2", answer="=HYPERLINK(\"x\")", needs_manual=True),
    ]
    out = build_csv(rubric, rows)
    parsed = list(csv.DictReader(io.StringIO(out)))
    assert len(parsed) == 2
    assert parsed[0]["answer"] == "Rayleigh scattering,\nwith a newline"
    assert parsed[0]["ai_score"] == "4"
    assert parsed[0]["final_score"] == "5"
    assert parsed[0]["max_score"] == "5"
    assert parsed[0]["reviewed"] == "yes"
    assert parsed[1]["answer"].startswith("'=")
    assert parsed[1]["ai_score"] == ""
    assert parsed[1]["needs_manual_grading"] == "yes"
    assert parsed[1]["reviewed"] == "no"
