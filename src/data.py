"""Load the UCI E. coli promoter dataset (Harley & Reynolds, 1987).

Each line of promoters.data is: class,name,sequence
  class    one of {+,-}  (+ = promoter)
  name     instance name
  sequence 57 nucleotides, lowercase {a,c,g,t}, positions -50..+7

106 instances, balanced 53/53. CC BY 4.0, DOI 10.24432/C5S01D.
"""

from __future__ import annotations

from pathlib import Path

import torch

BASES = "acgt"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
SEQ_LEN = 57

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "promoters.data"


def encode(seq: str) -> list[int]:
    """Map a lowercase DNA string to integer token ids."""
    return [BASE_TO_IDX[b] for b in seq]


def load_promoters(path: Path = DATA_FILE) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse promoters.data into (sequences, labels) tensors.

    Sequences come back as (n, SEQ_LEN) int64 token ids; labels as (n,) with
    1 = promoter, 0 = non-promoter. Whitespace and tabs in the raw sequence
    field are stripped, and the length is asserted so a malformed row fails
    loudly rather than silently truncating.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` to download it from the UCI "
            "repository (DOI 10.24432/C5S01D)."
        )

    seqs: list[list[int]] = []
    labels: list[int] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        cls, _name, sequence = line.split(",")
        sequence = sequence.strip().replace("\t", "").lower()
        if len(sequence) != SEQ_LEN:
            raise ValueError(f"expected {SEQ_LEN} bases, got {len(sequence)}: {sequence!r}")
        seqs.append(encode(sequence))
        labels.append(1 if cls == "+" else 0)

    return torch.tensor(seqs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
