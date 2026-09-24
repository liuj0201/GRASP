"""Deterministic artifacts, hashing, and cache provenance.

No cache is accepted merely because a filename exists.  A descriptor binds
the artifact to its exact sources, frozen configuration, fold, parameters,
and upstream artifact hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA = "da-cf-gop.cache.v1"
CACHE_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact",
        "config_sha256",
        "fold",
        "sources",
        "parameters",
        "upstream_sha256",
        "fingerprint",
    }
)
PATH_RECORD_KEYS = frozenset({"path", "kind", "bytes", "sha256"})


class ProvenanceError(ValueError):
    """Raised when an artifact cannot be proven to match its descriptor."""


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProvenanceError(f"non-finite JSON number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProvenanceError(f"JSON object key at {path} is not a string")
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ProvenanceError(f"non-finite array at {path}")
    elif isinstance(value, np.generic):
        _reject_nonfinite(value.item(), path)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON with sorted keys, UTF-8, and no non-finite numbers."""

    _reject_nonfinite(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Hash a file or a directory inventory without following hidden state."""

    item = Path(path).resolve()
    if item.is_file():
        return sha256_file(item)
    if not item.is_dir():
        raise ProvenanceError(f"source path does not exist: {item}")
    digest = hashlib.sha256()
    files = sorted((entry for entry in item.rglob("*") if entry.is_file()),
                   key=lambda entry: entry.relative_to(item).as_posix())
    for entry in files:
        relative = entry.relative_to(item).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(entry)))
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return resolved.as_posix()
    base = root.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as error:
        raise ProvenanceError(f"path escapes provenance root: {resolved}") from error


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def path_record(path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    item = Path(path).resolve()
    if not item.exists() or not (item.is_file() or item.is_dir()):
        raise ProvenanceError(f"cannot record missing path: {item}")
    base = Path(root).resolve() if root is not None else None
    return {
        "path": _safe_relative(item, base),
        "kind": "file" if item.is_file() else "directory",
        "bytes": _size(item),
        "sha256": sha256_path(item),
    }


def _record_path(record: Mapping[str, object], root: Path | None) -> Path:
    declared = Path(str(record["path"]))
    path = declared.resolve() if declared.is_absolute() else (
        (root / declared).resolve() if root is not None else declared.resolve()
    )
    if root is not None:
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ProvenanceError(f"recorded path escapes provenance root: {path}") from error
    return path


def verify_path_record(
    record: Mapping[str, object], *, root: str | os.PathLike[str] | None = None
) -> Path:
    if set(record) != PATH_RECORD_KEYS:
        raise ProvenanceError("path record schema changed")
    if record["kind"] not in {"file", "directory"}:
        raise ProvenanceError("path record kind is invalid")
    base = Path(root).resolve() if root is not None else None
    path = _record_path(record, base)
    if not path.exists():
        raise ProvenanceError(f"recorded path is missing: {path}")
    observed_kind = "file" if path.is_file() else "directory" if path.is_dir() else None
    if observed_kind != record["kind"]:
        raise ProvenanceError(f"recorded path kind changed: {path}")
    if _size(path) != int(record["bytes"]):
        raise ProvenanceError(f"recorded path size changed: {path}")
    if sha256_path(path) != record["sha256"]:
        raise ProvenanceError(f"recorded path hash changed: {path}")
    return path


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def write_json(path: str | os.PathLike[str], value: object) -> None:
    _atomic_write(Path(path), canonical_json_bytes(value))


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | os.PathLike[str]) -> object:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ProvenanceError(f"non-finite JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProvenanceError(f"invalid JSON: {path}") from error


def write_deterministic_npz(
    path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]
) -> None:
    """Write a pickle-free NPZ with sorted members and fixed ZIP metadata."""

    if not arrays:
        raise ProvenanceError("NPZ must contain at least one array")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ProvenanceError(f"invalid NPZ member name: {name!r}")
            array = np.asanyarray(arrays[name])
            if array.dtype.hasobject:
                raise ProvenanceError(f"object arrays are forbidden: {name}")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    _atomic_write(Path(path), output.getvalue())


def load_npz(
    path: str | os.PathLike[str], *, expected_sha256: str | None = None
) -> dict[str, np.ndarray]:
    item = Path(path)
    if expected_sha256 is not None and sha256_file(item) != expected_sha256:
        raise ProvenanceError(f"NPZ hash mismatch: {item}")
    try:
        with np.load(item, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ProvenanceError(f"unsafe or invalid NPZ: {item}") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ProvenanceError(f"object array found in NPZ: {item}")
    return arrays


def build_cache_descriptor(
    *,
    artifact_type: str,
    artifact: str | os.PathLike[str],
    config: object,
    sources: Mapping[str, str | os.PathLike[str]],
    fold: str | None = None,
    parameters: Mapping[str, object] | None = None,
    upstream_sha256: Mapping[str, str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build a self-fingerprinting cache descriptor."""

    if not artifact_type.strip() or not sources:
        raise ProvenanceError("artifact_type and at least one source are required")
    base = Path(root).resolve() if root is not None else None
    upstream = dict(sorted((upstream_sha256 or {}).items()))
    if any(not isinstance(value, str) or len(value) != 64 for value in upstream.values()):
        raise ProvenanceError("upstream hashes must be SHA-256 hex strings")
    body: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "artifact_type": artifact_type,
        "artifact": path_record(artifact, root=base),
        "config_sha256": sha256_json(config),
        "fold": fold,
        "sources": {
            name: path_record(path, root=base) for name, path in sorted(sources.items())
        },
        "parameters": dict(parameters or {}),
        "upstream_sha256": upstream,
    }
    body["fingerprint"] = sha256_json(body)
    return body


def validate_cache_descriptor(
    descriptor: Mapping[str, object],
    *,
    root: str | os.PathLike[str] | None = None,
    expected_config: object | None = None,
    expected_artifact_type: str | None = None,
    expected_fold: str | None = None,
    required_source_roles: Iterable[str] = (),
) -> Path:
    """Validate descriptor fingerprint and every current file/directory hash."""

    if set(descriptor) != CACHE_DESCRIPTOR_KEYS:
        raise ProvenanceError("cache descriptor schema changed")
    if descriptor["schema_version"] != CACHE_SCHEMA:
        raise ProvenanceError("cache descriptor version changed")
    declared_fingerprint = descriptor["fingerprint"]
    unsigned = {key: descriptor[key] for key in descriptor if key != "fingerprint"}
    if declared_fingerprint != sha256_json(unsigned):
        raise ProvenanceError("cache descriptor fingerprint mismatch")
    if expected_artifact_type is not None and descriptor["artifact_type"] != expected_artifact_type:
        raise ProvenanceError("cache artifact type mismatch")
    if expected_fold is not None and descriptor["fold"] != expected_fold:
        raise ProvenanceError("cache fold mismatch")
    if expected_config is not None and descriptor["config_sha256"] != sha256_json(expected_config):
        raise ProvenanceError("cache configuration mismatch")

    sources = descriptor["sources"]
    if not isinstance(sources, Mapping):
        raise ProvenanceError("cache sources must be an object")
    missing_roles = set(required_source_roles).difference(sources)
    if missing_roles:
        raise ProvenanceError(f"cache descriptor lacks sources: {sorted(missing_roles)}")
    base = Path(root).resolve() if root is not None else None
    for record in sources.values():
        if not isinstance(record, Mapping):
            raise ProvenanceError("cache source record must be an object")
        verify_path_record(record, root=base)
    artifact_record = descriptor["artifact"]
    if not isinstance(artifact_record, Mapping):
        raise ProvenanceError("cache artifact record must be an object")
    return verify_path_record(artifact_record, root=base)


def companion_descriptor_path(artifact: str | os.PathLike[str]) -> Path:
    return Path(f"{Path(artifact)}.descriptor.json")
