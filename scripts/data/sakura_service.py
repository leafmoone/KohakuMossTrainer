"""Start a step-controlled Sakura shard service from a TOML configuration."""

import argparse
from pathlib import Path

import tomllib
from sakuradatasetservice.cli import main as service_main
from sakuradatasetservice.manifest import parse_dataset_manifest_bytes


def arguments(config):
    source = config["source"]
    service = config["service"]
    kind = source["kind"]
    if kind == "local":
        provider = ["--local-root", source["local_root"]]
    elif kind == "modelscope":
        provider = ["--modelscope"]
    else:
        raise ValueError("source.kind must be 'local' or 'modelscope'")
    manifest = Path(service["manifest"]).expanduser().resolve()
    repo_id, revision = source["repo_id"], source["revision"]
    if manifest.exists():
        existing = parse_dataset_manifest_bytes(manifest.read_bytes())
        if (existing.source.repo_id, existing.source.revision) != (repo_id, revision):
            raise ValueError(
                "manifest source does not match configured repo_id/revision"
            )
        create = None
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        create = [
            "manifest",
            *provider,
            "--repo-id",
            repo_id,
            "--revision",
            revision,
            "--output",
            str(manifest),
        ]
    serve = [
        "serve",
        *provider,
        "--manifest",
        str(manifest),
        "--cache-root",
        service["cache_root"],
        "--state-dir",
        service["state_dir"],
        "--cache-high-bytes",
        str(service["cache_high_bytes"]),
        "--download-concurrency",
        str(service.get("download_concurrency", 1)),
        "--workers",
        "1",
        "--sampling",
        "balance",
        "--repeat",
        "--per-folder",
        str(service.get("per_folder", 5)),
        "--seed",
        str(service.get("seed", 0)),
    ]
    return create, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/sakura_service.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as handle:
        create, serve = arguments(tomllib.load(handle))
    if create:
        service_main(create)
    service_main(serve)


if __name__ == "__main__":
    main()
