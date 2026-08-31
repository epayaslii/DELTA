"""Action-name text embeddings for the crossmodal grounding branch.

DELTA uses DistilBERT. For the multimodal-SSL line we want text in the *same
space* as the visual features, so the default is the SigLIP2 text tower (pairs
with the ``vl3-siglip`` / ``siglip2`` visual backbones). ``distilbert`` is kept
for parity with the original paper.

Action names are turned into short prompts before encoding
(``"add_oil" -> "a photo of a person adding oil"``) via a small hand rule; pass
``raw=True`` to embed the bare label.
"""

from __future__ import annotations

import re

import numpy as np
import torch


def _humanize(label: str) -> str:
    words = re.sub(r"[_\-]+", " ", label).strip().lower()
    return f"a video of a person {words}" if words else "background"


def _prompts(names: list[str], raw: bool) -> list[str]:
    return list(names) if raw else [_humanize(n) for n in names]


def build_text_encoder(name: str = "siglip2", device: str = "cuda", dtype: str = "fp32"):
    name = name.lower()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    if name in ("siglip2", "siglip"):
        return _SigLIP2Text(device, torch_dtype)
    if name == "distilbert":
        return _DistilBERT(device, torch_dtype)
    raise ValueError(f"unknown text encoder {name!r}")


class _SigLIP2Text:
    dim = 1152

    def __init__(self, device, torch_dtype):
        from transformers import AutoModel, AutoTokenizer

        mid = "google/siglip2-so400m-patch14-384"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(mid)
        base = AutoModel.from_pretrained(mid, torch_dtype=torch_dtype).to(device).eval()
        self.model = base.text_model
        self.dim = int(base.config.text_config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**batch)
        emb = getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state[:, -1]
        return emb.float().cpu().numpy().astype(np.float32)


class _DistilBERT:
    dim = 768

    def __init__(self, device, torch_dtype):
        from transformers import AutoModel, AutoTokenizer

        mid = "distilbert-base-uncased"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModel.from_pretrained(mid, torch_dtype=torch_dtype).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return mean.float().cpu().numpy().astype(np.float32)


def encode_action_names(
    names: list[str],
    encoder: str = "siglip2",
    device: str = "cuda",
    raw: bool = False,
) -> np.ndarray:
    """Return (num_classes, D) embeddings aligned to ``names`` order (== mapping.txt)."""
    enc = build_text_encoder(encoder, device=device)
    return enc.encode(_prompts(names, raw))
