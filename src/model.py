"""A transformer classifier for E. coli promoter recognition.

The model embeds each nucleotide, prepends a learned CLS token, runs a small
transformer encoder, and classifies from the CLS representation. CLS pooling
(rather than mean pooling) suits a task where the signal is a localized motif
in a fixed-length window, which the -10 and -35 promoter boxes are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from src.data import BASES, SEQ_LEN


@dataclass
class Config:
    seq_len: int = SEQ_LEN
    vocab_size: int = len(BASES)
    d_model: int = 24
    n_heads: int = 2
    n_layers: int = 1
    d_ff: int = 48
    dropout: float = 0.3
    batch_size: int = 16
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42


class PromoterTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # A conv stem gives the model a motif-detector inductive bias before
        # attention. Promoter signals are short local boxes (-10, -35), so a
        # small receptive field over the embedded sequence is the right prior
        # and is what makes a ~100-example set learnable.
        # padding="same" keeps the conv output length equal to seq_len, so the
        # sequence fed to the transformer is exactly seq_len + 1 (with CLS) and
        # pos_embed matches it one-to-one. Any kernel size stays correct.
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=7, padding="same"),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len + 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.token_embed(x)  # (b, L, d)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)  # local motif features, length preserved
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if h.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"sequence length {h.size(1)} does not match pos_embed "
                f"{self.pos_embed.size(1)}; input length must equal Config.seq_len"
            )
        h = h + self.pos_embed
        h = self.encoder(h)
        logits: torch.Tensor = self.head(self.norm(h[:, 0, :]))
        return logits


def save_checkpoint(model: PromoterTransformer, cfg: Config, path: Path) -> None:
    """Write weights plus the config needed to rebuild the architecture.

    The state dict is snapshotted to CPU here so saving never moves the
    caller's model off its training device.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"config": asdict(cfg), "state_dict": cpu_state}, path)


def load_checkpoint(path: Path) -> tuple[PromoterTransformer, Config]:
    """Rebuild a model from a checkpoint written by save_checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**blob["config"])
    model = PromoterTransformer(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg
