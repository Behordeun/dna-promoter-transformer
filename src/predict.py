"""Classify a DNA sequence from a saved checkpoint.

Loads a model written by src.train_human (or any save_checkpoint call), encodes
one sequence, and prints the predicted class and the promoter probability. The
checkpoint carries its own Config, so the vocabulary and sequence length are
taken from the model it was trained with rather than guessed here.

Run:
    python -m src.predict --checkpoint checkpoints/human.pt "ACGT...251bp..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import Config, load_checkpoint

# Human path: ACGT plus a dedicated unknown id for N and other ambiguous codes.
# E. coli checkpoints use the 4-letter acgt vocabulary with no unknown slot.
_BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(_BASES)}


def encode(seq: str, cfg: Config) -> torch.Tensor:
    """Encode one sequence to a (1, seq_len) tensor using the checkpoint's vocab."""
    seq = seq.strip().upper()
    if len(seq) != cfg.seq_len:
        raise ValueError(f"sequence has {len(seq)} bases, model expects {cfg.seq_len}")
    unk = len(_BASES)  # only reachable when the model has a 5th (unknown) token
    ids = []
    for base in seq:
        idx = _BASE_TO_IDX.get(base, unk)
        if idx >= cfg.vocab_size:
            raise ValueError(
                f"base {base!r} needs an unknown token but this model's vocab is "
                f"{cfg.vocab_size}; it was trained on a strict ACGT alphabet"
            )
        ids.append(idx)
    return torch.tensor([ids], dtype=torch.long)


def predict(checkpoint: Path, seq: str) -> tuple[int, float]:
    """Return (predicted_class, promoter_probability) for one sequence."""
    model, cfg = load_checkpoint(checkpoint)
    x = encode(seq, cfg)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    promoter_prob = float(probs[1])
    return int(probs.argmax()), promoter_prob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("sequence", help="DNA sequence of the model's expected length")
    args = parser.parse_args()

    label, prob = predict(args.checkpoint, args.sequence)
    verdict = "promoter" if label == 1 else "non-promoter"
    print(f"prediction: {verdict}  (promoter probability {prob:.3f})")


if __name__ == "__main__":
    main()
