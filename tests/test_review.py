import csv
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import build_csv  # noqa: E402
from models import Criterion, GradedAnswer, Rubric  # noqa: E402
from review import (  # noqa: E402
    DEFAULT_RUBRIC_ROWS,
    results_to_table,
    rubric_from_table,
    table_to_graded,
)


def table(rows):
    return pd.DataFrame(rows, columns=["Criterion", "Description", "Max points"])


def test_default_rows_build_a_valid_rubric():
    rubric, errors = rubric_from_table("Why is the sky blue?", pd.DataFrame(DEFAULT_RUBRIC_ROWS))
    assert errors == [] and rubric.max_total == 5


def test_blank_rows_ignored_and_nan_handled():
    t = table(
        [
            ["Accuracy", None, 3.0],
            [None, None, None],  # untouched new row from the editor
            ["Clarity", "", float("nan")],
        ]
    )
    rubric, errors = rubric_from_table("q", t)
    assert rubric is None
    assert errors == ["Rubric row 3 (Clarity): max points must be greater than 0."]


def test_errors_collected():
    rubric, errors = rubric_from_table("  ", table([[None, "desc", 2.0], ["A", "", 1.0], ["a", "", 1.0]]))
    assert rubric is None
    assert "Enter the question." in errors
    assert "Rubric row 1: add a criterion name." in errors
    assert "Rubric criterion names must be unique." in errors


def test_empty_rubric_message():
    rubric, errors = rubric_from_table("q", table([[None, None, None]]))
    assert rubric is None and errors == ["Add at least one rubric criterion with points."]


def _rubric():
    return Rubric(question="q", criteria=[Criterion(name="A", max_points=3), Criterion(name="B", max_points=2)])


def test_results_table_prefills_final_but_not_reviewed():
    graded = [
        GradedAnswer(label="S1", answer="x", ai_score=4, ai_feedback="ok", ai_breakdown="A 3/3; B 1/2"),
        GradedAnswer(label="S2", answer="y", needs_manual=True, error="boom"),
    ]
    t = results_to_table(graded)
    assert t["Final score"].iloc[0] == 4 and pd.isna(t["Final score"].iloc[1])
    assert not t["Reviewed"].any()
    assert str(t["AI score"].dtype) == "float64"


def test_all_failures_still_make_a_valid_table():
    t = results_to_table([GradedAnswer(label="S", answer="x", needs_manual=True)])
    assert str(t["Final score"].dtype) == "float64" and pd.isna(t["Final score"].iloc[0])


def test_teacher_edits_survive_and_are_clamped():
    t = results_to_table([GradedAnswer(label="S1", answer="x", ai_score=4, ai_feedback="ai text")])
    t.loc[0, "Final score"] = 99  # impossible grade
    t.loc[0, "Reviewed"] = True
    t.loc[0, "Feedback"] = "teacher wrote this"
    t.loc[0, "Student"] = "Ayesha"
    [g] = table_to_graded(t, _rubric())
    assert g.final_score == 5  # clamped to rubric max
    assert g.ai_score == 4  # original AI suggestion kept for the audit trail
    assert g.reviewed is True and g.label == "Ayesha" and g.ai_feedback == "teacher wrote this"


def test_blank_label_and_missing_hidden_columns():
    t = pd.DataFrame({"Student": [None], "Answer": ["x"], "Final score": [2.0]})
    [g] = table_to_graded(t, _rubric())
    assert g.label == "Student 1" and g.final_score == 2.0
    assert g.needs_manual is False and g.reviewed is False and g.ai_score is None


def test_full_path_to_csv():
    rubric = _rubric()
    t = results_to_table(
        [
            GradedAnswer(label="S1", answer="=cmd()", ai_score=4, ai_feedback="fine", ai_breakdown="A 3/3; B 1/2"),
            GradedAnswer(label="S2", answer="z", needs_manual=True, error="x"),
        ]
    )
    t.loc[0, "Final score"] = 3.5
    t.loc[0, "Reviewed"] = True
    parsed = list(csv.DictReader(io.StringIO(build_csv(rubric, table_to_graded(t, rubric)))))
    assert parsed[0]["ai_score"] == "4" and parsed[0]["final_score"] == "3.5"
    assert parsed[0]["reviewed"] == "yes" and parsed[0]["answer"] == "'=cmd()"
    assert parsed[1]["final_score"] == "" and parsed[1]["needs_manual_grading"] == "yes"
