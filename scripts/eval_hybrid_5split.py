"""Score hybrid_ta variants on all 5 splits of 50Salads (coarse SigLIP2 features).

The aligner has no trained parameters, so "5 splits" = every video scored once, grouped
by the split whose test set contains it. Only the ordered transcript (from the GT
segments) goes in; frame labels are used for scoring only.

    python scripts/eval_hybrid_5split.py --feat-dir data/50salads/features_siglip2_coarse \
        --out docs/hybrid-ta-5split.json

Two protocols: 19-class (all mapping.txt classes, transcript incl. action_start/end) and
17-class (same alignment, but frames/classes of action_start/end are ignored in scoring).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delta.align import segmentation_report  # noqa: E402
from delta.align.similarity import l2norm  # noqa: E402
from hybrid_ta import HybridConfig, generate_pseudo_labels  # noqa: E402

STRIDE = 30                                    # features were extracted every 30 label frames
CONFIGS = {                                    # name -> HybridConfig | "naive"
    "naive-uniform": "naive",
    "dp":       HybridConfig(method="dp", branches="none"),
    "dp+A":     HybridConfig(method="dp", branches="a"),
    "asot":     HybridConfig(method="asot", branches="none"),
    "asot+A":   HybridConfig(method="asot", branches="a"),
    "asot+B":   HybridConfig(method="asot", branches="b"),
    "asot+A+B": HybridConfig(method="asot", branches="ab"),
    # settings picked by eye on split 1 earlier (docs/50salads-notes.md): splits 2-5 are a clean holdout for them
    "asot[r.5,a.1]":       HybridConfig(method="asot", branches="none", rho=0.5, alpha=0.1),
    "asot[r.5,a.1]+A":     HybridConfig(method="asot", branches="a", rho=0.5, alpha=0.1),
    "asot[r.5,a.1]+A[90,40]": HybridConfig(method="asot", branches="a", rho=0.5, alpha=0.1, radius=90, window=40),
    "asot+A[90,40]":       HybridConfig(method="asot", branches="a", radius=90, window=40),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/50salads"))
    ap.add_argument("--feat-dir", default=str(ROOT / "data/50salads/features_siglip2_coarse"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data, feat_dir = Path(a.data), Path(a.feat_dir)

    name2id = {l.split(" ", 1)[1].strip(): int(l.split()[0]) for l in open(data / "mapping.txt") if l.strip()}
    txt = l2 = np.load(feat_dir / "action_name_embeddings.npy").astype(np.float64)
    txt = l2norm(txt)
    split_of = {}
    for k in range(1, 6):
        for l in open(data / f"splits/test.split{k}.bundle"):
            if l.strip():
                split_of[l.strip().replace(".txt", "")] = k

    rows = {c: {"19": [], "17": [], "split": []} for c in CONFIGS}
    for vid in sorted(split_of):
        gt = np.array([name2id[l.strip()] for l in open(data / f"groundTruth/{vid}.txt") if l.strip()])
        f = np.load(feat_dir / f"{vid}.npy")
        f = (f.T if f.shape[0] == txt.shape[1] else f)[:len(gt)]
        fg = f[::STRIDE].astype(np.float64)                          # the real (non-filled) grid
        sim = l2norm(fg) @ txt.T                                     # (T', 19)
        transcript = [int(gt[0])] + [int(y) for p, y in zip(gt[:-1], gt[1:]) if y != p]
        Tg = len(fg)
        up = np.minimum(np.round(np.arange(len(gt)) / STRIDE).astype(int), Tg - 1)
        for name, cfg in CONFIGS.items():
            if cfg == "naive":
                ent = np.minimum((np.arange(len(gt)) * len(transcript)) // len(gt), len(transcript) - 1)
                pred = np.asarray(transcript)[ent]
            else:
                y = generate_pseudo_labels(sim[:, transcript], transcript, fg, cfg).y_star
                pred = y[up]
            rows[name]["19"].append(segmentation_report(pred, gt))
            rows[name]["17"].append(segmentation_report(pred, gt, ignore={17, 18}))
            rows[name]["split"].append(split_of[vid])

    out = {}
    print(f"{len(split_of)} videos\n")
    for proto in ("19", "17"):
        print(f"== {proto}-class protocol  (mean over videos; F1@50 in %)")
        print(f"{'config':24s} {'MoF':>6s} {'MoC':>6s} {'edit':>6s} {'F1@50':>6s}   MoC per split 1..5")
        for name in CONFIGS:
            r = rows[name][proto]; sp = np.array(rows[name]["split"])
            m = {k: float(np.mean([x[k] for x in r])) for k in ("MoF", "MoC", "edit", "F1@50")}
            per = [float(np.mean([x["MoC"] for x, s in zip(r, sp) if s == k])) for k in range(1, 6)]
            out.setdefault(proto, {})[name] = {**m, "MoC_per_split": per}
            print(f"{name:24s} {m['MoF']:6.3f} {m['MoC']:6.3f} {m['edit']:6.1f} {m['F1@50']:6.1f}   " + " ".join(f"{p:.3f}" for p in per))
        print()
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
