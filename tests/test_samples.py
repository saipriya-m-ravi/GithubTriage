import json
from pathlib import Path

import pytest

from githubtriage.models import Classification, TriageState

SAMPLES_DIR = Path(__file__).parent.parent / "samples" / "issues"
MANIFEST = SAMPLES_DIR / "expected.json"
SAMPLE_FILES = sorted(p for p in SAMPLES_DIR.glob("*.json") if p != MANIFEST)


@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_is_valid_state(path):
    TriageState.model_validate_json(path.read_text(encoding="utf-8-sig"))


def test_manifest_covers_every_sample_with_a_valid_type():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    valid_types = set(Classification.model_fields["type"].annotation.__args__)

    assert set(expected) == {p.name for p in SAMPLE_FILES}
    for name, entry in expected.items():
        assert entry["type"] in valid_types, name
