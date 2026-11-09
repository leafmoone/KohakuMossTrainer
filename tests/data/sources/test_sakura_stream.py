"""The service stream consumes leased TARs without interpreting rounds as epochs."""

import io
import json
import tarfile
from threading import Event, Thread

import pytest
from PIL import Image
from sakuradatasetservice import (
    CacheQuota,
    DataServiceLimits,
    DataServiceServer,
    DataServiceSessionIdentity,
    DatasetSourceIdentity,
    DataSupplyService,
    LocalDatasetProvider,
    ShardCache,
)

from kohakumoss.training.step import LayoutTransform
from kohakumoss_data import DataConfig, build_loader, data_problems


def test_service_stream_reads_multiple_shards(tmp_path):
    source = tmp_path / "source"
    for folder in ("a", "b[1]"):
        (source / folder).mkdir(parents=True)
        with tarfile.open(source / folder / "0.tar", "w") as tar:
            image = io.BytesIO()
            Image.new("RGB", (32, 32), "red").save(image, format="PNG")
            metadata = json.dumps({"tags": {"character": [folder]}}).encode()
            for name, payload in (("0.png", image.getvalue()), ("0.json", metadata)):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    provider = LocalDatasetProvider(source)
    manifest = provider.fetch_manifest(
        DatasetSourceIdentity(repo_id="local/test", revision="v1")
    )
    cache = ShardCache(
        tmp_path / "cache", manifest, provider, CacheQuota(0, 1024 * 1024)
    )
    service = DataSupplyService(
        manifest,
        cache,
        tmp_path / "queue",
        tmp_path / "lock",
        DataServiceSessionIdentity(manifest.dataset_id, 1),
        DataServiceLimits(1, 1, 1, 1),
        sampling="balance",
        per_folder=5,
        repeat=True,
    )
    server = DataServiceServer(
        service, tmp_path / "data.sock", request_timeout_seconds=3.0
    )
    stop, ready = Event(), Event()
    thread = Thread(
        target=server.serve, args=(stop,), kwargs={"ready_callback": ready.set}
    )
    thread.start()
    try:
        assert ready.wait(5)
        cfg = DataConfig(
            datasets=[
                {"source": "sakura_service", "socket": str(tmp_path / "data.sock")}
            ],
            batching="count",
            batch_size=2,
            target_area=32 * 32,
            latent_cache="off",
            text_cache="off",
            num_workers=0,
            text_drop=0,
            caption={
                "template": "multichoice",
                "weights": {
                    "tags": 1,
                    "nl2": 0,
                    "nl3": 0,
                    "long_no_names": 0,
                    "long_names": 0,
                    "short_vibes": 0,
                },
            },
        )
        loader = build_loader(cfg, transform=LayoutTransform())
        it = iter(loader)
        assert {next(it).prompts[0], next(it).prompts[0]} == {"a", "b[1]"}
        current = next(it).prompts[0]
        assert current in {"a", "b[1]"}
        state = loader.state_dict()
        it.close()
        loader._close_handles()
        resumed = build_loader(cfg)
        resumed.load_state_dict(state)
        assert next(iter(resumed)).prompts[0] == current
        with pytest.raises(ValueError, match="seed"):
            resumed.load_state_dict({"seed": 99, "start": 0})
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()


def test_service_config_rejects_incompatible_data_settings():
    cfg = DataConfig(datasets=[{"source": "sakura_service"}])
    problems = " ".join(data_problems(cfg))
    assert "NUM_WORKERS=0" in problems
    assert "LATENT_CACHE=TEXT_CACHE='off'" in problems
    assert "socket" in problems
