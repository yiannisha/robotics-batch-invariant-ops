import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compress_preserved_rollouts.py"


def write_json(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return payload


def test_compression_is_byte_exact_and_manifested(tmp_path: Path) -> None:
    write_json(tmp_path / "reset_pairing_audit.json", {"fully_paired": True})
    sources = {
        tmp_path / "native_singleton/rollouts/one.json": write_json(
            tmp_path / "native_singleton/rollouts/one.json", {"episodes": [{"success": True}]},
        ),
        tmp_path / "native_dynamic/rollouts/two.json": write_json(
            tmp_path / "native_dynamic/rollouts/two.json", {"episodes": [{"success": False}]},
        ),
    }

    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--root", str(tmp_path),
        "--allow-fully-paired", "--expected-files", "2",
    ], cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    for source, payload in sources.items():
        compressed = source.with_suffix(".json.gz")
        assert not source.exists()
        assert gzip.decompress(compressed.read_bytes()) == payload
    manifest = json.loads((tmp_path / "rollout_compression_manifest.json").read_text())
    assert manifest["files"] == 2
    assert manifest["all_decompressed_sha256_verified"] is True
    by_source = {item["original_path"]: item for item in manifest["records"]}
    for source, payload in sources.items():
        assert by_source[str(source)]["original_sha256"] == hashlib.sha256(payload).hexdigest()


def test_compression_refuses_nonterminal_evidence_without_touching_source(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "reset_pairing_audit.json", {"fully_paired": False})
    source = tmp_path / "native_singleton/rollouts/one.json"
    payload = write_json(source, {"episodes": [{"success": True}]})

    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--root", str(tmp_path),
        "--allow-fully-paired", "--expected-files", "1",
    ], cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert source.read_bytes() == payload
    assert not source.with_suffix(".json.gz").exists()
    assert not (tmp_path / "rollout_compression_manifest.json").exists()


def test_compression_refuses_wrong_cardinality_before_writing(tmp_path: Path) -> None:
    write_json(tmp_path / "reset_pairing_audit.json", {"fully_paired": True})
    source = tmp_path / "native_singleton/rollouts/one.json"
    payload = write_json(source, {"episodes": [{"success": True}]})

    completed = subprocess.run([
        sys.executable, str(SCRIPT), "--root", str(tmp_path),
        "--allow-fully-paired", "--expected-files", "2",
    ], cwd=ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert source.read_bytes() == payload
    assert not source.with_suffix(".json.gz").exists()
    assert not (tmp_path / "rollout_compression_manifest.json").exists()
