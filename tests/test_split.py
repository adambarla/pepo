"""Tests for ensemble data splitting (disjoint vs overlapping subsampling)."""

from __future__ import annotations

import numpy as np
from datasets import Dataset

from pepo.data.manager import DataManager
from pepo.model.deppo import DEPPOModel


def _make_dataset(n: int = 100) -> Dataset:
    return Dataset.from_list([{"id": i} for i in range(n)])


def test_disjoint_split_basic() -> None:
    """Disjoint: each model gets ~N/L samples, no overlap between shards."""
    data = _make_dataset(100)
    dm = object.__new__(DataManager)
    dm.n_splits = 4
    dm.seed = 42
    dm.split_mode = "disjoint"
    setattr(dm, "_sort_by_length", lambda ds: ds)

    dm._split_train(data)

    assert len(dm.train_datasets) == 4
    sizes = [len(dm.train_datasets[i]) for i in range(4)]
    assert sizes == [25, 25, 25, 25]

    shard_sets = [set(dm.train_datasets[i]["id"]) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert shard_sets[i].isdisjoint(shard_sets[j])


def test_disjoint_split_reproducible() -> None:
    """Same seed produces identical disjoint splits."""
    data = _make_dataset(100)
    dm1 = object.__new__(DataManager)
    dm1.n_splits = 4
    dm1.seed = 42
    dm1.split_mode = "disjoint"
    setattr(dm1, "_sort_by_length", lambda ds: ds)
    dm1._split_train(data)

    dm2 = object.__new__(DataManager)
    dm2.n_splits = 4
    dm2.seed = 42
    dm2.split_mode = "disjoint"
    setattr(dm2, "_sort_by_length", lambda ds: ds)
    dm2._split_train(data)

    for i in range(4):
        assert list(dm1.train_datasets[i]["id"]) == list(dm2.train_datasets[i]["id"])


def test_overlap_subsample_split_subset_size() -> None:
    """Overlapping subsampling: each model gets N/L distinct samples;
    members may overlap."""
    data = _make_dataset(100)
    dm = object.__new__(DataManager)
    dm.n_splits = 4
    dm.seed = 42
    dm.split_mode = "overlap_subsample"
    setattr(dm, "_sort_by_length", lambda ds: ds)

    dm._split_train(data)

    assert len(dm.train_datasets) == 4
    for i in range(4):
        assert len(dm.train_datasets[i]) == 100 // dm.n_splits
        ids = dm.train_datasets[i]["id"]
        assert len(ids) == len(set(ids))


def test_overlap_subsample_has_overlap() -> None:
    """Overlapping subsampling: different models share samples, but each
    member has no duplicates."""
    np.random.seed(0)
    data = _make_dataset(1000)
    dm = object.__new__(DataManager)
    dm.n_splits = 4
    dm.seed = 42
    dm.split_mode = "overlap_subsample"
    setattr(dm, "_sort_by_length", lambda ds: ds)

    dm._split_train(data)

    sets = [set(dm.train_datasets[i]["id"]) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert len(sets[i] & sets[j]) > 0, (
                f"Models {i} and {j} have zero overlap in overlap_subsample"
            )
            assert sets[i] != sets[j], (
                f"Models {i} and {j} are identical in overlap_subsample"
            )


def test_overlap_subsample_samples_from_original() -> None:
    """Overlapping subsampling: every sample originates from the original dataset."""
    data = _make_dataset(100)
    original_ids = set(data["id"])
    dm = object.__new__(DataManager)
    dm.n_splits = 4
    dm.seed = 42
    dm.split_mode = "overlap_subsample"
    setattr(dm, "_sort_by_length", lambda ds: ds)

    dm._split_train(data)

    for i in range(4):
        for sample_id in dm.train_datasets[i]["id"]:
            assert sample_id in original_ids


def test_overlap_subsample_reproducible() -> None:
    """Same seed produces identical overlapping subsets."""
    data = _make_dataset(100)
    dm1 = object.__new__(DataManager)
    dm1.n_splits = 4
    dm1.seed = 42
    dm1.split_mode = "overlap_subsample"
    setattr(dm1, "_sort_by_length", lambda ds: ds)
    dm1._split_train(data)

    dm2 = object.__new__(DataManager)
    dm2.n_splits = 4
    dm2.seed = 42
    dm2.split_mode = "overlap_subsample"
    setattr(dm2, "_sort_by_length", lambda ds: ds)
    dm2._split_train(data)

    for i in range(4):
        assert list(dm1.train_datasets[i]["id"]) == list(dm2.train_datasets[i]["id"])


def test_single_model_ignores_split_mode() -> None:
    """When n_splits <= 1, split_mode is irrelevant — all data goes to model 0."""
    data = _make_dataset(100)
    for mode in ("disjoint", "overlap_subsample"):
        dm = object.__new__(DataManager)
        dm.n_splits = 1
        dm.seed = 42
        dm.split_mode = mode
        setattr(dm, "_sort_by_length", lambda ds: ds)
        dm._split_train(data)
        assert len(dm.train_datasets) == 1
        assert list(dm.train_datasets[0]["id"]) == list(range(100))


def test_unknown_split_mode_raises() -> None:
    """Unknown split_mode raises ValueError."""
    data = _make_dataset(100)
    dm = object.__new__(DataManager)
    dm.n_splits = 4
    dm.seed = 42
    dm.split_mode = "overlap_50"
    setattr(dm, "_sort_by_length", lambda ds: ds)

    import pytest

    with pytest.raises(ValueError, match="Unknown split_mode"):
        dm._split_train(data)


def test_get_name_overlap_subsample_suffix() -> None:
    """Overlapping subsampling models get a -overlap_subsample suffix in
    their Hub name."""
    model = object.__new__(DEPPOModel)
    model.model_id = "org/MyModel"
    model.alpha = 0.1
    model.beta = 0.1
    model._num_models = 4
    model.split_mode = "overlap_subsample"

    name = model.get_name(model_idx=2, epoch=8)
    assert name == "MyModel-a0.1-b0.1-L4-overlap_subsample-l2-e8"


def test_get_name_disjoint_no_suffix() -> None:
    """Disjoint models do NOT get a suffix (backward-compatible naming)."""
    model = object.__new__(DEPPOModel)
    model.model_id = "org/MyModel"
    model.alpha = 0.1
    model.beta = 0.1
    model._num_models = 4
    model.split_mode = "disjoint"

    name = model.get_name(model_idx=1, epoch=3)
    assert name == "MyModel-a0.1-b0.1-L4-l1-e3"


def test_get_name_all_combinations() -> None:
    """All combinations of model_idx/epoch produce correct names."""
    model = object.__new__(DEPPOModel)
    model.model_id = "org/M"
    model.alpha = 0.1
    model.beta = 0.1
    model._num_models = 4
    model.split_mode = "overlap_subsample"

    assert model.get_name() == "M-a0.1-b0.1-L4-overlap_subsample"
    assert model.get_name(model_idx=0) == "M-a0.1-b0.1-L4-overlap_subsample-l0"
    assert model.get_name(epoch=5) == "M-a0.1-b0.1-L4-overlap_subsample-e5"
    assert (
        model.get_name(model_idx=3, epoch=16)
        == "M-a0.1-b0.1-L4-overlap_subsample-l3-e16"
    )
