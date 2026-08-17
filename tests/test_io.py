import json

from cgps.io import write_result


def test_result_writer_creates_parent_and_valid_json(tmp_path) -> None:
    destination = write_result(tmp_path / "nested" / "result.json", {"score": 1.0})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"score": 1.0}
    assert not destination.with_suffix(".json.tmp").exists()
