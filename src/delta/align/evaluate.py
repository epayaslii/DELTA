"""Evaluate a transcript->video aligner over a dataset split.

A *similarity provider* maps a video id -> an (N, T) transcript x frame matrix.
Swap providers to compare: naive-uniform, a frozen VLM
(`delta.align.similarity`), a vendored baseline (ATBA), etc. Scoring is always
`Y*` vs held-out ground truth via `delta.align.segmentation_report`.

CLI:
    python -m delta.align.evaluate --config configs/50salads.yaml --provider naive --split 1
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from delta.data import ActionSegDataset
from delta.data.datasets import DATASET_DEFAULTS
from delta.data.stats import naive_uniform_labeling
from delta.align import segmentation_report, align_dp, segments


# --------------------------------------------------------------------------------------
# similarity providers
# --------------------------------------------------------------------------------------


def provider_oracle_blocky(ds: ActionSegDataset, vid: str, jitter: float = 0.0, seed: int = 0):
    """(N, T) similarity that is 1.0 on each entry's TRUE block (+ optional noise).
    Not a real method -- an upper bound / aligner sanity check on real transcript
    shapes (N up to ~26, repeated classes)."""
    rec = ds.record(vid)
    T = rec.num_label_frames
    N = len(rec.transcript)
    rng = np.random.default_rng(seed)
    s = rng.normal(0, jitter, size=(N, T)) if jitter else np.zeros((N, T))
    # walk the GT segments; entry index advances per segment
    for n, (_, a, b) in enumerate(segments(rec.frame_labels)):
        s[n, a:b] += 1.0
    return s, rec


def provider_vlm(ds: ActionSegDataset, vid: str, frame_dir: str, class_emb: np.ndarray,
                 temperature: float | None = None):
    """(N, T) cosine similarity between frozen VLM frame embeddings
    (`<frame_dir>/<vid>.npy`, shape (T, D) or (D, T)) and per-class text
    embeddings (C, D). Requires features extracted with a VLM backbone."""
    from delta.align.similarity import similarity_matrix, transcript_text_embeddings

    rec = ds.record(vid)
    f = np.load(f"{frame_dir}/{vid}.npy")
    if f.shape[0] != rec.num_label_frames and f.shape[1] == rec.num_label_frames:
        f = f.T
    txt = transcript_text_embeddings(rec.transcript, class_emb)
    return similarity_matrix(txt, f, temperature=temperature), rec


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------


def evaluate(ds: ActionSegDataset, ids, provider, ignore=None, transition_penalty=0.0,
             **prov_kw):
    """provider: "naive" | a callable (ds, vid, **prov_kw) -> (s (N,T), record)."""
    ignore = ignore or set()
    rows = []
    for vid in ids:
        rec = ds.record(vid)
        if provider == "naive":
            y = naive_uniform_labeling(rec.transcript, rec.num_label_frames)
        else:
            s, rec = provider(ds, vid, **prov_kw)
            y = align_dp(s, rec.transcript, transition_penalty=transition_penalty).y_star
        r = segmentation_report(y, rec.frame_labels, ignore=ignore)
        r["video_id"] = vid
        rows.append(r)
    keys = [k for k in rows[0] if k != "video_id"]
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    return mean, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--provider", default="naive", choices=["naive", "oracle", "vlm"])
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--subset", default="test", choices=["train", "test"])
    p.add_argument("--frame-dir", help="VLM feature dir (provider=vlm)")
    p.add_argument("--class-emb", help="action_name_embeddings.npy (provider=vlm)")
    p.add_argument("--temperature", type=float)
    p.add_argument("--tp", type=float, default=0.0, help="transition penalty")
    p.add_argument("--ignore-startend", action="store_true")
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(open(args.config))
    ds = ActionSegDataset(cfg["dataset"]["name"], cfg["dataset"]["root"])
    ids = ds.split(args.split, args.subset)

    ignore = set()
    if args.ignore_startend:
        ignore = {i for i, n in enumerate(ds.idx_to_name) if n in ("action_start", "action_end")}

    if args.provider == "naive":
        mean, _ = evaluate(ds, ids, "naive", ignore=ignore)
    elif args.provider == "oracle":
        mean, _ = evaluate(ds, ids, provider_oracle_blocky, ignore=ignore,
                           transition_penalty=args.tp)
    else:
        class_emb = np.load(args.class_emb)
        mean, _ = evaluate(ds, ids, provider_vlm, ignore=ignore,
                           transition_penalty=args.tp,
                           frame_dir=args.frame_dir, class_emb=class_emb,
                           temperature=args.temperature)
    print(f"[{args.provider}] {cfg['dataset']['name']} split{args.split} {args.subset} "
          f"(n={len(ids)})")
    print(json.dumps({k: round(v, 3) for k, v in mean.items()}, indent=2))


if __name__ == "__main__":
    main()
