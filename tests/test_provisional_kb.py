import os
import json
import uuid
from app import save_provisional_kb_entry


def test_save_provisional_creates_file_and_entry():
    category = f"temp_tests_{uuid.uuid4().hex[:6]}"
    topic = f"Provisional_{uuid.uuid4().hex[:6]}"
    code = "print('hello')\n"
    metrics = {"score": 0.0}
    analysis_id = "test_analysis"

    kb_path = os.path.join("Knowledge_base", f"{category}.json")
    # Ensure clean state
    if os.path.exists(kb_path):
        os.remove(kb_path)

    save_provisional_kb_entry(
        category, topic, code, metrics,
        description="test", analysis_id=analysis_id
    )

    assert os.path.exists(kb_path)
    with open(kb_path, "r") as fh:
        data = json.load(fh)
    assert any(entry.get("topic") == topic for entry in data)

    # Cleanup: remove created KB file
    os.remove(kb_path)
