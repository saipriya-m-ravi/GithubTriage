"""The eval answer key. Each case mirrors a real pydantic issue seen 2026-11-10."""

import pytest

from githubtriage.evaluation.labels import Excluded, GroundTruth, ground_truth

REPORTER = {"login": "reporter", "type": "User"}
MAINTAINER = {"login": "maintainer", "type": "User"}
BOT = {"login": "labeler[bot]", "type": "Bot"}


def issue(*labels: str, association: str = "NONE") -> dict:
    return {
        "user": {"login": "reporter"},
        "author_association": association,
        "labels": [{"name": name} for name in labels],
    }


def labeled(name: str, actor: dict = REPORTER) -> dict:
    return {"event": "labeled", "label": {"name": name}, "actor": actor}


def unlabeled(name: str, actor: dict = MAINTAINER) -> dict:
    return {"event": "unlabeled", "label": {"name": name}, "actor": actor}


# The bug template: the REPORTER's account applies both labels.
TEMPLATE_BUG = [labeled("bug V2"), labeled("pending")]


def test_maintainer_applied_label_is_gold():
    # pydantic #14237: bug V2 applied by a maintainer.
    truth = ground_truth(issue("bug V2"), [labeled("bug V2", MAINTAINER)])
    assert truth == GroundTruth(
        type="bug", label="bug V2", trust="gold", relabelled=False
    )


def test_template_bug_reviewed_by_a_maintainer_is_gold():
    # #14234: the template's labels, then a maintainer removed "pending".
    truth = ground_truth(issue("bug V2"), [*TEMPLATE_BUG, unlabeled("pending")])
    assert truth.trust == "gold"


def test_still_pending_is_excluded():
    # #14211: nobody has triaged it, so the label is only the reporter's guess (D1).
    assert ground_truth(issue("bug V2", "pending"), TEMPLATE_BUG) == Excluded(
        reason="unreviewed: still pending"
    )


def test_relabelled_by_a_maintainer_is_gold_and_flagged_as_hard():
    # #12580: filed with the bug form, a maintainer changed it to question.
    events = [
        *TEMPLATE_BUG,
        unlabeled("bug V2"),
        unlabeled("pending"),
        labeled("question", MAINTAINER),
    ]
    truth = ground_truth(issue("question"), events)
    assert truth == GroundTruth(
        type="question",
        label="question",
        trust="gold",
        relabelled=True,
        reporter_label="bug V2",  # the form they picked: for the baseline only
    )


def test_reporter_only_feature_request_is_silver():
    # #14063: the feature form adds no "pending", and nobody else touched it —
    # we can't tell whether a maintainer agreed (D2).
    truth = ground_truth(issue("feature request"), [labeled("feature request")])
    assert truth.trust == "silver"
    assert truth.type == "feature"


def test_issue_filed_by_a_maintainer_is_gold():
    truth = ground_truth(
        issue("feature request", association="MEMBER"), [labeled("feature request")]
    )
    assert truth.trust == "gold"


def test_a_bots_label_is_not_a_human_judgement():
    truth = ground_truth(issue("documentation"), [labeled("documentation", BOT)])
    assert truth.trust == "silver"


def test_several_of_our_labels_are_excluded():
    # Ambiguous: which one is the answer? (D3)
    truth = ground_truth(issue("bug V2", "documentation"), [])
    assert truth == Excluded(reason="several of our labels: bug V2, documentation")


@pytest.mark.parametrize("labels", [(), ("bug V1",), ("help wanted",)])
def test_issues_without_one_of_our_labels_are_excluded(labels):
    # bug V1 is deliberately unmapped: V1-era issues describe a different codebase.
    assert ground_truth(issue(*labels), []) == Excluded(reason="none of our labels")


def test_other_labels_alongside_ours_are_ignored():
    truth = ground_truth(issue("documentation", "good first issue"), [])
    assert truth.type == "docs"


def test_ghost_actor_does_not_count_as_a_maintainer():
    # Deleted accounts appear with actor: null.
    events = [{"event": "labeled", "label": {"name": "question"}, "actor": None}]
    assert ground_truth(issue("question"), events).trust == "silver"
