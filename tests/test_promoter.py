"""Tests: real data parses correctly, model wiring, and CV clears baselines."""

from __future__ import annotations

import torch

from src.data import SEQ_LEN, encode, load_promoters
from src.model import Config, PromoterTransformer, load_checkpoint, save_checkpoint
from src.predict import predict
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


def test_conv_stem_preserves_length_across_kernels() -> None:
    # The pos_embed sizing must stay correct no matter the seq_len; same-padding
    # guarantees the transformer input is seq_len + 1 by construction.
    for seq_len in (57, 100, 251):
        cfg = Config(seq_len=seq_len, vocab_size=4)
        model = PromoterTransformer(cfg)
        x = torch.randint(0, 4, (2, seq_len))
        assert model(x).shape == (2, 2)


def test_checkpoint_round_trip_predicts_same(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = Config()  # fast E. coli config, no download
    model = PromoterTransformer(cfg)
    model.eval()
    ckpt = tmp_path / "m.pt"
    save_checkpoint(model, cfg, ckpt)

    reloaded, reloaded_cfg = load_checkpoint(ckpt)
    assert reloaded_cfg.seq_len == cfg.seq_len
    assert reloaded_cfg.vocab_size == cfg.vocab_size

    seq = "a" * cfg.seq_len
    label, prob = predict(ckpt, seq)
    assert label in (0, 1)
    assert 0.0 <= prob <= 1.0

    # The reloaded model must produce identical logits to the original.
    x = torch.tensor([encode(seq)], dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)


def test_cross_validation_beats_naive_baselines() -> None:
    # Use the real default config and 10-fold protocol. The model should clear
    # the ID3 baseline (19/106) comfortably; the full run reaches 7/106.
    cfg = Config()
    errors, total, _ = cross_validate(cfg, k=10)
    assert total == 106
    assert errors < 19, f"expected fewer than 19 errors (ID3 baseline), got {errors}"
