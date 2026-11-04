import logging
import types
from datetime import UTC, datetime
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel

from githubtriage.checkpointing import CHECKPOINT_CLASSES, make_serializer
from githubtriage.models import (
    Classification,
    Completeness,
    Decision,
    FieldCheck,
    Issue,
    TriageState,
)


def models_in(annotation) -> set[type[BaseModel]]:
    """Every Pydantic model inside a type annotation, e.g. `Completeness | None`."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {annotation}
    if get_origin(annotation) in (Union, types.UnionType, Annotated, list, dict):
        return set().union(*(models_in(arg) for arg in get_args(annotation)))
    return set()


def reachable_models(model: type[BaseModel]) -> set[type[BaseModel]]:
    """Models used by `model`'s fields, recursively (Completeness -> FieldCheck)."""
    found: set[type[BaseModel]] = set()
    stack = [model]
    while stack:
        for field in stack.pop().model_fields.values():
            for m in models_in(field.annotation) - found:
                found.add(m)
                stack.append(m)
    return found


def test_allowlist_covers_every_model_in_the_state():
    # Fails if a new model is added to TriageState but not to CHECKPOINT_CLASSES —
    # which in strict mode would silently come back from a checkpoint as a dict.
    # TriageState itself is included: the input passed to invoke() is stored too.
    assert set(CHECKPOINT_CLASSES) == reachable_models(TriageState) | {TriageState}


def test_state_values_round_trip_with_their_types(caplog):
    ok = FieldCheck(status="ok", note="")
    now = datetime.now(UTC)
    values = [
        Issue(title="t", body="b", author="a", updated_at=now),
        Classification(type="bug", confidence=0.9, reasoning="x"),
        Completeness(version=ok, repro_steps=ok, logs=ok),
        Decision(action="approve", approver="m", decided_at=now),
    ]
    serde = make_serializer()

    with caplog.at_level(logging.WARNING):
        for value in values:
            restored = serde.loads_typed(serde.dumps_typed(value))
            assert type(restored) is type(value)
            assert restored == value

    # No "unregistered type" / "Blocked deserialization" warnings from LangGraph.
    assert not [r for r in caplog.records if "deserializ" in r.getMessage().lower()]
