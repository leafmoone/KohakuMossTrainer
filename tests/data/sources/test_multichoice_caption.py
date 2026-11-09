"""Weighted captions from nested sidecar metadata."""

import json

import numpy as np
import pytest

from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import build_text, caption_set
from kohakumoss_data.caption import build_template
from kohakumoss_data.registry import SOURCES
from tests._util.encode_fns import ByteCodes
from tests.data import fixtures


@pytest.fixture
def sample(tmp_path):
    (tmp_path / "one.png").write_bytes(fixtures.png_bytes(64, 64, 1))
    data = {
        "nsfw": "nsfw",
        "tags": {
            "general": ["1girl", "blue_hair"],
            "artist": ["artist_a"],
            "copyright": ["series"],
            "character": [],
        },
        "rating": "explicit",
        "year": "year 2015",
        "quality": "normal",
        "anime_completeness": "polished",
        "anime_classification": "illustration",
        "captions": {"nl2": "two", "nl3": "three"},
        "multicaptions": {
            "long_no_names": "long",
            "long_names": "named",
            "short": "short",
            "vibes": "vivid",
        },
    }
    (tmp_path / "one.json").write_text(json.dumps(data), encoding="utf-8")
    return next(SOURCES.get("folder")(str(tmp_path)).scan())


NAMES = ("tags", "nl2", "nl3", "long_no_names", "long_names", "short_vibes")


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "tags",
            "nsfw, 1girl, blue_hair, artist_a, series, explicit, year 2015, normal, polished, illustration",
        ),
        ("nl2", "two"),
        ("nl3", "three"),
        ("long_no_names", "long"),
        ("long_names", "named"),
        ("short_vibes", "short, vivid"),
    ],
)
def test_each_option(sample, name, expected):
    weights = {key: float(key == name) for key in NAMES}
    template = build_template({"template": "multichoice", "weights": weights})
    assert template.render(sample, np.random.default_rng(3)) == expected


def test_missing_fields_and_empty_values(sample):
    sample.json = {
        "nsfw": "",
        "tags": {"general": ["", "1girl", None]},
        "multicaptions": {"short": "", "vibes": " mood "},
    }
    sample.rating = None
    for name, expected in (("tags", "1girl"), ("nl2", ""), ("short_vibes", "mood")):
        template = build_template(
            {"template": "multichoice", "weights": {k: int(k == name) for k in NAMES}}
        )
        assert template.render(sample, np.random.default_rng(0)) == expected


def test_weighted_sampling(sample):
    template = build_template(
        {
            "template": "multichoice",
            "weights": {k: {"nl2": 1, "nl3": 3}.get(k, 0) for k in NAMES},
        }
    )
    rng = np.random.default_rng(19)
    outputs = [template.render(sample, rng) for _ in range(4000)]
    assert set(outputs) == {"two", "three"}
    assert 0.70 < outputs.count("three") / len(outputs) < 0.80


def test_default_weighted_sampling(sample):
    template = build_template({"template": "multichoice"})
    rng = np.random.default_rng(23)
    outputs = [template.render(sample, rng) for _ in range(12000)]
    for text, weight in (
        ("two", 1),
        ("three", 1),
        ("long", 1),
        ("named", 3),
        ("short, vivid", 1),
    ):
        assert outputs.count(text) / len(outputs) == pytest.approx(
            weight / 12, abs=0.015
        )
    assert sum(text.startswith("nsfw, ") for text in outputs) / len(
        outputs
    ) == pytest.approx(5 / 12, abs=0.015)


@pytest.mark.parametrize(
    "weights", [{"nl2": -1}, {"nl2": float("nan")}, {"typo": 1}, {k: 0 for k in NAMES}]
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        build_template({"template": "multichoice", "weights": weights})


def test_dataset_and_offline_cache(sample, tmp_path):
    encoder = ByteCodes(dim=32)
    config = DataConfig(
        datasets=[{"source": "folder", "path": str(tmp_path)}],
        caption={"template": "multichoice", "weights": {k: 1 for k in NAMES}},
        caption_variants=32,
        latent_cache="off",
        text_cache="offline",
        cache_dir=str(tmp_path / "cache"),
        target_area=64 * 64,
        max_tokens=1024,
        text_drop=0,
        text_encoder_id=encoder.encoder_id,
    )
    dataset = MossDataset(config)
    expected = set(caption_set(dataset))
    assert {"two", "three", "long", "named", "short, vivid"} <= expected
    assert any(text.startswith("nsfw, 1girl, blue_hair") for text in expected)
    assert build_text(dataset, encoder) == len(expected)
    reopened = MossDataset(config)
    assert set(caption_set(reopened)) == expected
    assert all(reopened.texts.key(text) in reopened.texts.store for text in expected)
