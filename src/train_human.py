"""Train the conv+transformer on the human non-TATA promoter dataset (GPU).

This is the genome-scale task: 27,097 training and 9,034 test sequences of
251 bp. It trains on the Apple MPS GPU when available (falls back to CUDA or
CPU), carves a validation slice off the training split for early stopping, and
reports accuracy, precision, recall and F1 on the held-out official test set.

Run:
    python -m src.train_human
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_human import SEQ_LEN, VOCAB_SIZE, load_human_promoters
from src.device import get_device
from src.model import Config, PromoterTransformer, save_checkpoint


def _metrics(preds: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, ...]], device: torch.device
) -> dict[str, float]:
    model.eval()
    preds, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        ys.append(yb)
    return _metrics(torch.cat(preds), torch.cat(ys))


def train_human(
    cfg: Config,
    val_frac: float = 0.1,
    patience: int = 3,
    checkpoint_dir: Path | None = None,
) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}")

    (x_train, y_train), (x_test, y_test) = load_human_promoters()

    n_val = int(len(x_train) * val_frac)
    perm = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(cfg.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(x_train[tr_idx], y_train[tr_idx]),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(x_train[val_idx], y_train[val_idx]), batch_size=256, num_workers=0
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, num_workers=0)

    model = PromoterTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state = None
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        val = _evaluate(model, val_loader, device)["acc"]
        print(f"epoch {epoch:2d}/{cfg.epochs}  val_acc={val:.4f}")
        if val > best_val:
            best_val, best_state, stale = val, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (best val_acc={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = _evaluate(model, test_loader, device)
    print(
        f"\nHeld-out test: acc={test['acc']:.4f}  precision={test['precision']:.4f}  "
        f"recall={test['recall']:.4f}  f1={test['f1']:.4f}"
    )

    if checkpoint_dir is not None:
        ckpt = checkpoint_dir / "human.pt"
        save_checkpoint(model.cpu(), cfg, ckpt)
        (checkpoint_dir / "metrics.json").write_text(json.dumps(test, indent=2))
        print(f"saved checkpoint to {ckpt} and metrics to {checkpoint_dir / 'metrics.json'}")

    return test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="where to save human.pt and metrics.json",
    )
    args = parser.parse_args()

    cfg = Config(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.layers,
        d_ff=128,
        dropout=0.2,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=5e-4,
        weight_decay=1e-3,
    )
    train_human(cfg, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
