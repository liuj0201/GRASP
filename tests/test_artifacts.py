from da_cf_gop.artifacts import read_jsonl, write_jsonl_gz
from da_cf_gop.provenance import sha256_file


def test_gzip_jsonl_is_deterministic(tmp_path):
    rows = [{"b": 2, "a": 1}, {"b": 4, "a": 3}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    write_jsonl_gz(first, rows)
    write_jsonl_gz(second, rows)
    assert sha256_file(first) == sha256_file(second)
    assert read_jsonl(first) == rows
