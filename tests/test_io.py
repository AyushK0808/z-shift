import json
from pathlib import Path

from spatial_ingestion.reconstruction._io import write_json


def test_write_json_writes_indented_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    write_json(path, {"a": 1, "b": 2})
    content = path.read_text(encoding="utf-8")
    assert json.loads(content) == {"a": 1, "b": 2}
    assert content.endswith("\n")
