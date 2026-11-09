"""The launcher binds the configured source to a fixed service manifest."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sakuradatasetservice import DatasetSourceIdentity, LocalDatasetProvider
from sakuradatasetservice.manifest import write_dataset_manifest


def test_config_launcher_creates_manifest_and_rejects_source_mismatch(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/data/sakura_service.py"
    spec = importlib.util.spec_from_file_location("sakura_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.tar").write_bytes(b"x")
    config = {
        "source": {
            "kind": "local",
            "local_root": str(source),
            "repo_id": "local/x",
            "revision": "v1",
        },
        "service": {
            "manifest": str(tmp_path / "manifest.json"),
            "cache_root": str(tmp_path / "cache"),
            "state_dir": str(tmp_path / "state"),
            "cache_high_bytes": 10000,
        },
    }
    create, serve = module.arguments(config)
    assert create[0] == "manifest"
    assert serve[serve.index("--per-folder") + 1] == "5"
    manifest = LocalDatasetProvider(source).fetch_manifest(
        DatasetSourceIdentity(repo_id="local/x", revision="v1")
    )
    write_dataset_manifest(manifest, tmp_path / "manifest.json")
    assert module.arguments(config)[0] is None
    config["source"]["revision"] = "v2"
    with pytest.raises(ValueError, match="does not match"):
        module.arguments(config)


def test_launcher_starts_real_service_from_toml(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.tar").write_bytes(b"test")
    root = Path(__file__).resolve().parents[3]
    config = tmp_path / "config.toml"

    def quoted(path):
        return json.dumps(str(path))

    config.write_text(
        "[source]\nkind='local'\nrepo_id='local/x'\nrevision='v1'\n"
        f"local_root={quoted(source)}\n"
        "[service]\ncache_high_bytes=1048576\n"
        f"manifest={quoted(tmp_path / 'manifest.json')}\n"
        f"cache_root={quoted(tmp_path / 'cache')}\n"
        f"state_dir={quoted(tmp_path / 'state')}\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "scripts/data/sakura_service.py"),
            "--config",
            str(config),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        socket = tmp_path / "state/data.sock"
        while (
            not socket.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert socket.exists(), (
            process.stderr.read().decode()
            if process.poll() is not None
            else "service did not start"
        )
        assert (tmp_path / "manifest.json").is_file()
    finally:
        process.terminate()
        process.communicate(timeout=10)
