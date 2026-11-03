"""Load precomputed descriptions through the original dataset-attribute API."""
from functools import lru_cache
import json
from pathlib import Path
import re

DESCRIPTION_SETS = ("gpt4_general", "gpt4_tsp_paper")
DEFAULT_DESCRIPTION_SET = "gpt4_tsp_paper"
DATASETS = ("dtd", "imagenet", "caltech101", "eurosat", "fgvc", "food101",
            "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101")
_ROOT = Path(__file__).resolve().parent
_selected_set = DEFAULT_DESCRIPTION_SET


def configure_descriptions(description_set=DEFAULT_DESCRIPTION_SET):
    """Select the description set before constructing or evaluating a model."""
    if description_set not in DESCRIPTION_SETS:
        raise ValueError(f"Unknown description set {description_set!r}; expected {DESCRIPTION_SETS}")
    global _selected_set
    _selected_set = description_set


def normalize_classname(name):
    return re.sub(r"[\s_]+", " ", name.strip().lower())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def description_path(dataset, description_set=None):
    group = description_set or _selected_set
    if group not in DESCRIPTION_SETS or dataset not in DATASETS:
        raise ValueError(f"Unknown description set/dataset: {group}/{dataset}")
    return _ROOT / group / f"{dataset}.json"


class _ClassDescriptions(dict):
    def __missing__(self, name):
        if isinstance(name, str):
            key = normalize_classname(name)
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)
        raise KeyError(f"Missing class description: {name!r}")


@lru_cache(maxsize=None)
def _load(dataset, group):
    path = description_path(dataset, group)
    with path.open(encoding="utf-8-sig") as stream:
        raw = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Expected a nonempty class-to-description object: {path}")
    candidates = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError(f"Class names and descriptions must be nonempty strings: {path}")
        candidates.setdefault(normalize_classname(key), []).append(key)
    result = _ClassDescriptions()
    for key, source_keys in candidates.items():
        if len(source_keys) > 1:
            raise ValueError(f"Ambiguous normalized class {key!r} in {path}")
        source_key = source_keys[0]
        result[key] = raw[source_key]
    print(f"[descriptions] set={group} dataset={dataset} file={path.name}", flush=True)
    return result


def get_descriptions(dataset, description_set=None):
    return _load(dataset, description_set or _selected_set)


def __getattr__(dataset):
    if dataset not in DATASETS:
        raise AttributeError(dataset)
    return get_descriptions(dataset)
