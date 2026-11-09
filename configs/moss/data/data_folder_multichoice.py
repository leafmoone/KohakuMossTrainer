"""Folder with same-stem JSON sidecars: select one of six caption formats.

Set the dataset path; see docs/concepts/data.md#multi-choice-json-captions.
"""

DATASETS = [{"source": "folder", "path": "path/to/dataset"}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {
    "template": "multichoice",
    "weights": {
        "tags": 5,
        "nl2": 1,
        "nl3": 1,
        "long_no_names": 1,
        "long_names": 3,
        "short_vibes": 1,
    },
}
CAPTION_VARIANTS = 16
