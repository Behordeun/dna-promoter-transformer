"""Tests: real data parses correctly, model wiring, and CV clears baselines."""

from __future__ import annotations

import torch

from src.data import SEQ_LEN, encode, load_promoters
from src.model import Config, PromoterTransformer
from src.train import _stratified_folds, cross_validate


def test_encode_maps_lowercase_bases() -> None:
    assert encode("acgt") == [0, 1, 2, 3]


def test_dataset_is_the_real_balanced_106() -> None:
    x, y = load_promoters()
    assert x.shape == (106, SEQ_LEN)
    assert y.shape == (106,)
    assert int(y.sum().item()) == 53  # 53 promoters, 53 non-promoters


def test_folds_are_stratified_and_cover_every_sample() -> None:
    _, y = load_promoters()
    rng = torch.Generator().manual_seed(0)
    folds = _stratified_folds(y, k=10, rng=rng)
    covered = torch.cat(folds).sort().values
    assert covered.tolist() == list(range(106))
    # Each fold holds both classes.
    for f in folds:
        assert 0 < int(y[f].sum().item()) < f.numel()


def test_forward_returns_two_logits() -> None:
    cfg = Config()
    model = PromoterTransformer(cfg)
    x = torch.randint(0, 4, (8, SEQ_LEN))
    assert model(x).shape == (8, 2)


def test_cross_validation_beats_naive_baselines() -> None:
    # Use the real default config and 10-fold protocol. The model should clear
    # the ID3 baseline (19/106) comfortably; the full run reaches 9/106.
    cfg = Config()
    errors, total, _ = cross_validate(cfg, k=10)
    assert total == 106
    assert errors < 19, f"expected fewer than 19 errors (ID3 baseline), got {errors}"
