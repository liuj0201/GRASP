"""Frozen artifact layout and deterministic JSONL/CSV serialization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .provenance import canonical_json_bytes, sha256_file, write_json


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    @classmethod
    def from_path(cls, value: str | os.PathLike[str]) -> "ArtifactLayout":
        return cls(Path(value).resolve())

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def summaries(self) -> Path:
        return self.root / "summaries"

    def create(self) -> None:
        for path in (self.manifests, self.cache, self.evaluation, self.runtime, self.summaries):
            path.mkdir(parents=True, exist_ok=True)


def _atomic_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        candidate = Path(temporary)
        if candidate.exists():
            candidate.unlink()


def write_jsonl_gz(
    path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]
) -> str:
    """Write canonical JSONL in a gzip stream with a fixed timestamp."""

    uncompressed = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
        stream.write(uncompressed)
    destination = Path(path)
    _atomic_bytes(destination, output.getvalue())
    return sha256_file(destination)


def write_jsonl(
    path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]
) -> str:
    """Write uncompressed canonical JSONL atomically."""

    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    destination = Path(path)
    _atomic_bytes(destination, payload)
    return sha256_file(destination)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}: {source}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {source}")
            rows.append(value)
    return rows


def write_csv(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> str:
    if not rows:
        raise ValueError("refusing to write a headerless empty CSV")
    columns = list(fieldnames or sorted({key for row in rows for key in row}))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    destination = Path(path)
    _atomic_bytes(destination, buffer.getvalue().encode("utf-8"))
    return sha256_file(destination)


def write_stage_summary(
    path: str | os.PathLike[str],
    *,
    stage: str,
    config_sha256: str,
    training_speakers: Sequence[str],
    source_files: Mapping[str, str],
    models: Mapping[str, str],
    manifests: Mapping[str, str],
    upstream_artifacts: Mapping[str, str],
    exclusions: Mapping[str, int],
    outputs: Mapping[str, str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": "da-cf-gop.stage-summary.v1",
        "stage": str(stage),
        "config_sha256": str(config_sha256),
        "training_speakers": sorted(set(str(value) for value in training_speakers)),
        "source_files": dict(sorted(source_files.items())),
        "models": dict(sorted(models.items())),
        "manifests": dict(sorted(manifests.items())),
        "upstream_artifacts": dict(sorted(upstream_artifacts.items())),
        "exclusions": dict(sorted((str(key), int(value)) for key, value in exclusions.items())),
        "outputs": dict(sorted(outputs.items())),
        "details": dict(details or {}),
    }
    write_json(path, summary)
    return summary
