"""Load the human non-TATA promoter dataset (genomic-benchmarks).

Real human promoter classification: 36,131 sequences of 251 bp, split by the
dataset authors into 27,097 train and 9,034 test. Two classes, positive =
promoter. Source: Grešová et al. (2023), "Genomic benchmarks: a collection of
datasets for genomic sequence classification", BMC Genomic Data 24:25.

This is the genome-scale task; src/data.py keeps the small E. coli set as a
fast sanity check. First run downloads ~and caches under the package's data
directory.
"""

from __future__ import annotations

import torch

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
# N and other ambiguous codes map to a dedicated pad/unknown id.
UNK_IDX = len(BASES)
VOCAB_SIZE = len(BASES) + 1
SEQ_LEN = 251
DATASET = "human_nontata_promoters"


def _encode(seq: str) -> list[int]:
    return [BASE_TO_IDX.get(b, UNK_IDX) for b in seq.upper()]


def _load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    from genomic_benchmarks.loc2seq import download_dataset

    path = download_dataset(DATASET, version=0)
    base = path / split
    seqs: list[list[int]] = []
    labels: list[int] = []
    for label, cls in enumerate(sorted(p.name for p in base.iterdir() if p.is_dir())):
        for f in (base / cls).glob("*.txt"):
            seq = f.read_text().strip()
            if len(seq) != SEQ_LEN:
                continue
            seqs.append(_encode(seq))
            labels.append(label)
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def load_human_promoters() -> tuple[
    tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]:
    """Return ((x_train, y_train), (x_test, y_test)) using the official split."""
    return _load_split("train"), _load_split("test")
