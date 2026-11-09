"""W&B sample images keep the exact prompt that generated each image."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from kohakumoss.logging import sinks
from kohakumoss.training.callbacks.evaluate import SampleGrid
from kohakumoss.training.state import TrainState

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


@pytest.mark.parametrize("columns", [0, 2])
def test_sample_prompts_are_attached_to_images(tmp_path, monkeypatch, columns):
    captured = []
    monkeypatch.setattr(
        sinks,
        "wandb",
        SimpleNamespace(Image=lambda image, caption=None: (image, caption)),
    )
    logger = object.__new__(sinks.WandbLogger)
    logger.run = SimpleNamespace(
        log=lambda payload, step: captured.append((payload, step))
    )
    prompts = ["first, full prompt", "second long prompt", "third prompt"]
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
    callback = SampleGrid(
        200,
        lambda **kw: images,
        {"prompts": prompts},
        str(tmp_path),
        logger,
        torch.nn.Linear(1, 1),
        (lambda: None, lambda: None),
        columns=columns,
    )
    callback.on_step_end(TrainState(step=0), None)
    records, step = captured[0]
    captions = [caption for _, caption in records["samples"]]
    assert step == 0
    if columns:
        assert len(captions) == 1
        assert all(prompt in captions[0] for prompt in prompts)
    else:
        assert captions == prompts


def test_original_references_are_paired_with_sample_prompts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        sinks,
        "wandb",
        SimpleNamespace(
            Image=lambda image, caption=None: (image, caption),
            Table=lambda columns, data: SimpleNamespace(columns=columns, data=data),
        ),
    )
    logger = object.__new__(sinks.WandbLogger)
    logger.run = SimpleNamespace(
        log=lambda payload, step: calls.append((payload, step))
    )
    prompts = ["first prompt", "second prompt"]
    originals = [str(tmp_path / "first.png"), str(tmp_path / "second.png")]
    Image.new("RGB", (16, 8)).save(originals[0])
    Image.new("RGB", (8, 16)).save(originals[1])
    callback = SampleGrid(
        200,
        lambda **kw: [Image.new("RGB", (8, 8)) for _ in prompts],
        {"prompts": prompts},
        str(tmp_path),
        logger,
        torch.nn.Linear(1, 1),
        (lambda: None, lambda: None),
        references=originals,
    )
    callback.on_step_end(TrainState(step=0), None)
    assert calls[1][1] == 0
    table = calls[1][0]["sample_references"]
    assert table.columns == ["sample", "reference", "prompt"]
    assert [row[1][0] for row in table.data] == originals
    assert [row[2] for row in table.data] == prompts
    gallery = calls[1][0]["sample_references_images"]
    assert [caption for _, caption in gallery] == [
        f"Sample {i + 1}: {prompt}" for i, prompt in enumerate(prompts)
    ]
    assert len({image.size for image, _ in gallery}) == 1


def test_reference_paths_are_only_used_for_matching_prompts(tmp_path):
    import json

    from kohakumoss.training.build.loop import sample_reference_paths

    root = tmp_path / "dataset"
    (root / "training").mkdir(parents=True)
    images = [root / "a.png", root / "b.png"]
    for image in images:
        image.touch()
    (root / "sample-prompts.json").write_text(json.dumps(["a", "b"]))
    (root / "sample-provenance.json").write_text(
        json.dumps(
            [{"index": i, "image": str(image)} for i, image in enumerate(images)]
        )
    )
    k = {
        "DATASETS": [{"source": "folder", "path": str(root / "training")}],
        "SAMPLE_PROMPTS": ["a", "b"],
    }
    assert sample_reference_paths(k) == [str(p) for p in images]
    assert sample_reference_paths({**k, "SAMPLE_PROMPTS": ["b", "a"]}) == []
