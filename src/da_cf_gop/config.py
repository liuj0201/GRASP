"""Frozen configuration loading and path resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "frozen_v1.json"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("schema_version") != "da-cf-gop.config.v1":
        raise ValueError(f"unsupported configuration schema: {value.get('schema_version')!r}")
    resolved = dict(value)
    resolved["_config_path"] = str(source)
    resolved["_config_hash"] = config_hash(value)
    paths = dict(value["paths"])
    base = source.parent.parent
    resolved["paths"] = {
        key: str((base / raw).resolve()) if not Path(raw).is_absolute() else str(Path(raw).resolve())
        for key, raw in paths.items()
    }
    if "severity" in value:
        severity = dict(value["severity"])
        for key in ("audited_manifest", "audited_logits_cache"):
            raw = severity[key]
            severity[key] = str(
                (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            )
        severity["frozen_scores"] = {
            key: str(
                (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
            )
            for key, raw in severity["frozen_scores"].items()
        }
        resolved["severity"] = severity
    return resolved
