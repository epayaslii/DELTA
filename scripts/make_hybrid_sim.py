"""Offline step for `--ta_source hybrid`: per-video frame x class similarity files.

    python scripts/make_hybrid_sim.py --feat-dir data/50salads/features_siglip2_coarse \
        --text-emb data/50salads/features_siglip2_coarse/action_name_embeddings.npy \
        --out-dir data/50salads/hybrid_sim [--keep-feat]

Reads `<feat-dir>/<video>.npy` (T, D) VLM frame features and the (C, D) class-name
text embeddings (row i <-> class id i of mapping.txt); writes `<out-dir>/<video>.npz`
with `sim` (T, C) cosine similarity and, with --keep-feat, `feat` (T, D) fp16 for the
relational branch. Uses no ground-truth labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_ta.similarity import cosine_similarity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--text-emb", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-feat", action="store_true")
    a = ap.parse_args()
    txt = torch.from_numpy(np.load(a.text_emb)).float()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.feat_dir).glob("*.npy") if p.name != Path(a.text_emb).name)
    for p in files:
        f = torch.from_numpy(np.load(p)).float()
        if f.shape[1] != txt.shape[1]:
            f = f.T                                       # some dumps are (D, T)
        sim = cosine_similarity(f[None], txt)[0].numpy().astype(np.float32)
        payload = {"sim": sim}
        if a.keep_feat:
            payload["feat"] = f.numpy().astype(np.float16)
        np.savez(out / f"{p.stem}.npz", **payload)
    print(f"wrote {len(files)} files -> {out}  (sim shape example: {sim.shape})")


if __name__ == "__main__":
    main()
