"""Train and evaluate the promoter transformer with stratified k-fold CV.

With only 106 examples a single train/test split is noise, so evaluation is
k-fold cross-validation: every sequence is held out exactly once, and the
reported number is the pooled error over all folds. That matches the
leave-one-out protocol the dataset's authors used, so the result sits next to
the published baselines on the same footing:

    KBANN       4/106 errors   (Towell, Shavlik & Noordewier, 1990)
    backprop    8/106
    nearest-nbr 13/106
    ID3        19/106

Run:
    python -m src.train
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from src.data import load_promoters
from src.model import Config, PromoterTransformer


def _stratified_folds(
    y: torch.Tensor, k: int, rng: torch.Generator
) -> list[torch.Tensor]:
    """Return k index tensors, each a fold, with class balance preserved."""
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = torch.where(y == cls)[0]
        idx = idx[torch.randperm(idx.size(0), generator=rng)]
        for position, sample in enumerate(idx.tolist()):
            folds[position % k].append(sample)
    return [torch.tensor(sorted(f)) for f in folds]


def _train_one(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    cfg: Config,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    n = x_tr.size(0)
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[batch]), y_tr[batch])
            loss.backward()
            optimizer.step()


def cross_validate(cfg: Config, k: int = 10) -> tuple[int, int, float]:
    """Run k-fold CV. Returns (errors, total, accuracy)."""
    torch.manual_seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)
    x, y = load_promoters()
    folds = _stratified_folds(y, k, rng)

    errors = 0
    total = 0
    for i, test_idx in enumerate(folds, start=1):
        train_idx = torch.cat([f for j, f in enumerate(folds) if j != i - 1])
        model = PromoterTransformer(cfg)
        _train_one(model, x[train_idx], y[train_idx], cfg)

        model.eval()
        with torch.no_grad():
            preds = model(x[test_idx]).argmax(dim=1)
        fold_err = int((preds != y[test_idx]).sum().item())
        errors += fold_err
        total += test_idx.size(0)
        print(f"fold {i:2d}/{k}  n={test_idx.size(0):2d}  errors={fold_err}")

    acc = 1.0 - errors / total
    return errors, total, acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    errors, total, acc = cross_validate(cfg, k=args.folds)
    print(f"\nCross-validated: {errors}/{total} errors, accuracy {acc:.3f}")
    print("Published baselines on this set: KBANN 4/106, backprop 8/106, "
          "nearest-neighbour 13/106, ID3 19/106.")


if __name__ == "__main__":
    main()
