"""Choosing which file in a Hub repo is the dataset.

A dataset repo carries metadata beside its data, and every metadata file has
a data-ish extension. Taking the first match by listing order therefore picks
``dataset_dict.json`` -- whose whole content is ``{"splits": ["train"]}`` --
loads it as the dataset, and reports the mapped column missing. Which is
true, and says nothing about the real problem.
"""

from surogate_eval.benchmarks.backends.custom_eval_backend import (
    _pick_dataset_file,
)

# The exact listing a Studio-uploaded dataset produces, in the order the Hub
# returns it.
REAL_REPO = [
    "_hub_actions/stats-worker.yaml",
    "dataset_dict.json",
    "train/dataset.parquet",
]


def test_metadata_is_never_mistaken_for_data():
    assert _pick_dataset_file(REAL_REPO, "train") == "train/dataset.parquet"


def test_it_still_works_without_a_split():
    # `split` narrows the choice; it is not required to avoid the metadata.
    assert _pick_dataset_file(REAL_REPO) == "train/dataset.parquet"


def test_the_requested_split_wins():
    # A repo with several splits otherwise loads whichever sorts first and
    # scores the wrong rows without saying so.
    paths = [
        "dataset_dict.json",
        "test/data.parquet",
        "train/data.parquet",
        "validation/data.parquet",
    ]
    assert _pick_dataset_file(paths, "train") == "train/data.parquet"
    assert _pick_dataset_file(paths, "validation") == "validation/data.parquet"


def test_parquet_is_preferred_over_looser_formats():
    paths = ["train/data.csv", "train/data.parquet", "train/data.jsonl"]
    assert _pick_dataset_file(paths, "train") == "train/data.parquet"


def test_an_unconverted_upload_still_loads():
    # Nothing published as parquet yet: the raw upload is all there is.
    paths = ["dataset_dict.json", "train/rows.jsonl"]
    assert _pick_dataset_file(paths, "train") == "train/rows.jsonl"


def test_nothing_usable_returns_none():
    # So the caller can name what it did find, rather than downloading a
    # YAML and failing three layers later.
    assert _pick_dataset_file(["_hub_actions/stats-worker.yaml"], "train") is None
    assert _pick_dataset_file([], "train") is None


def test_a_split_that_is_absent_falls_back_rather_than_failing():
    # Asking for a split the repo does not have still finds the data; the
    # caller's own split handling reports the mismatch, if any.
    assert _pick_dataset_file(REAL_REPO, "test") == "train/dataset.parquet"


def test_a_repo_holding_only_metadata_yields_nothing():
    """The case the extension ranking alone does not cover.

    A dataset whose data has not landed yet still lists its metadata, and
    every metadata file has a data-ish extension. Without excluding them by
    name, `dataset_dict.json` is the best remaining candidate and gets
    downloaded and loaded as the dataset -- the exact failure this guards.
    """
    assert _pick_dataset_file(["dataset_dict.json"], "train") is None
    assert _pick_dataset_file(["dataset_dict.json", "dataset_info.json"]) is None


def test_metadata_loses_to_data_of_the_same_kind():
    # Both `.json`; only the name tells them apart.
    paths = ["dataset_info.json", "rows.json"]
    assert _pick_dataset_file(paths) == "rows.json"
