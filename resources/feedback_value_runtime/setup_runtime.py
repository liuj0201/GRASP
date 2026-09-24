"""Download and verify pinned CPU runtime assets; no package installation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_HASH = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
LLAMA_TAG = "b10964"
LLAMA_COMMIT = "b29c606e28a01b1bc8c1351026a0fa6e616bf6c4"
LLAMA_NAME = "llama-b10964-bin-win-cpu-x64.zip"
LLAMA_HASH = "917f39c076402c421224824607397af20f53625a60defc20e8dd22446bf4c5d7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, expected_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected_hash:
        print(f"Verified existing {path.name}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "GRASP-experiment-runtime/1.0"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as target:
        total = int(source.headers.get("Content-Length", 0))
        count = 0
        next_report = 64 * 1024 * 1024
        while block := source.read(4 * 1024 * 1024):
            target.write(block)
            count += len(block)
            if count >= next_report:
                print(f"{path.name}: {count / 1e6:.1f}/{total / 1e6:.1f} MB; {time.monotonic() - start:.1f} s", flush=True)
                next_report += 64 * 1024 * 1024
    actual = sha256(partial)
    if actual != expected_hash:
        raise RuntimeError(f"SHA256 mismatch for {path.name}: {actual}")
    partial.replace(path)
    print(f"Downloaded and verified {path.name}: {actual}", flush=True)


def main() -> None:
    model_url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_NAME}"
    binary_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_NAME}"
    model_path = ROOT / "models" / MODEL_NAME
    archive_path = ROOT / "downloads" / LLAMA_NAME
    download(binary_url, archive_path, LLAMA_HASH)
    binary_dir = ROOT / "bin"
    binary_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            target = (binary_dir / entry.filename).resolve()
            if not target.is_relative_to(binary_dir.resolve()):
                raise RuntimeError("Unsafe archive path")
        archive.extractall(binary_dir)
    executables = list(binary_dir.rglob("llama-server.exe"))
    if len(executables) != 1:
        raise RuntimeError(f"Expected one llama-server.exe, found {executables}")
    executable = executables[0]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    version = subprocess.run([str(executable), "--version"], capture_output=True, text=True, creationflags=flags, check=True)
    help_result = subprocess.run([str(executable), "--help"], capture_output=True, text=True, creationflags=flags, check=True)
    (ROOT / "llama_server_help.txt").write_text(help_result.stdout + help_result.stderr, encoding="utf-8")
    print(version.stdout + version.stderr, flush=True)
    download(model_url, model_path, MODEL_HASH)
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"repository": MODEL_REPO, "revision": MODEL_REVISION, "filename": MODEL_NAME,
                  "source_url": model_url, "path": str(model_path), "sha256": MODEL_HASH,
                  "size_bytes": model_path.stat().st_size, "quantization": "Q4_K_M"},
        "runtime": {"repository": "ggml-org/llama.cpp", "release": "v0.4.1", "binary_tag": LLAMA_TAG,
                    "commit": LLAMA_COMMIT, "source_url": binary_url, "archive_sha256": LLAMA_HASH,
                    "executable": str(executable), "executable_sha256": sha256(executable),
                    "version_output": version.stdout + version.stderr, "platform": "win-cpu-x64"},
        "suggested_server_args": [str(executable), "-m", str(model_path), "--host", "127.0.0.1", "--port", "8097",
                                  "--device", "none", "--n-gpu-layers", "0", "--threads", "2", "--threads-batch", "2",
                                  "--parallel", "1", "--ctx-size", "4096", "--temp", "0", "--seed", "42",
                                  "--no-cont-batching", "--cache-ram", "0"],
        "note": "--parallel 1 controls simultaneous requests; --batch-size is token prompt batching, not request batch size."
    }
    (ROOT / "runtime_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {ROOT / 'runtime_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
