"""Sanity-check a backbone's text<->image alignment BEFORE a full extraction.

We already lost a run to this: the siglip2 backbone was reading pre-projection
features, so ``cos(action_text, frame)`` came out ~-0.02 (unrelated) and every
alignment collapsed. A few minutes here saves hours of GPU time.

Three checks, on a handful of real 50Salads frames with known labels:

  1. text<->image cosines land in a plausible range (not ~0, not all identical)
  2. frames vary across time (a near-constant encoder carries no signal)
  3. the GT action beats a random other action more often than chance

Usage
-----
    python scripts/verify_backbone_alignment.py --backbone videollama3
    python scripts/verify_backbone_alignment.py --backbone siglip2 --device cuda
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="videollama3")
    p.add_argument("--text-encoder", default="siglip2",
                   help="text tower to pair with (must share the backbone's space)")
    p.add_argument("--config", default="configs/50salads.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--videos", type=int, default=3, help="how many videos to probe")
    p.add_argument("--frames", type=int, default=60, help="frames sampled per video")
    args = p.parse_args(argv)

    import yaml
    from delta.data import ActionSegDataset
    from delta.features.backbones import build_backbone
    from delta.features.text_encoder import encode_action_names
    from delta.features.video_io import VideoReader

    cfg = yaml.safe_load(open(args.config))
    ds = ActionSegDataset(cfg["dataset"]["name"], cfg["dataset"]["root"],
                          video_dir=cfg["dataset"].get("video_dir"))
    names = ds.action_names()

    print(f"[text] encoding {len(names)} action names with {args.text_encoder}")
    txt = encode_action_names(names, encoder=args.text_encoder, device=args.device)

    print(f"[vision] building {args.backbone}")
    bb = build_backbone(args.backbone, device=args.device, dtype=args.dtype, batch_size=16)
    print(f"        dim={bb.dim}  (text dim={txt.shape[1]})")
    if bb.dim != txt.shape[1]:
        print("  !! DIM MISMATCH -- these two towers cannot be compared directly.")
        print("     Either pair a different text encoder, or use this backbone for")
        print("     visual structure only and get semantics from generated captions.")
        return 1

    def l2(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    all_ti, all_ff, hits, total = [], [], 0, 0
    for vid in ds.split(1, "test")[: args.videos]:
        rec = ds.record(vid)
        r = VideoReader(str(rec.video_path))
        idx = np.linspace(0, min(r.num_frames, rec.num_label_frames) - 1,
                          args.frames).astype(int)
        frames = r.get_batch(idx)
        r.close()
        f = l2(bb.encode(frames))
        gt = np.asarray(rec.frame_labels)[idx]

        sim = l2(txt) @ f.T                                  # (C, n)
        all_ti.append(sim)
        all_ff.append(f @ f.T)

        # does the GT action beat a random wrong action?
        rng = np.random.default_rng(0)
        for j, g in enumerate(gt):
            other = rng.choice([c for c in range(len(names)) if c != g])
            hits += sim[g, j] > sim[other, j]
            total += 1
        print(f"  {vid}: ti-cos mean {sim.mean():+.3f}  "
              f"range [{sim.min():+.3f}, {sim.max():+.3f}]")

    ti = np.concatenate([s.ravel() for s in all_ti])
    ff = np.concatenate([m[np.triu_indices_from(m, 1)] for m in all_ff])
    pairwise = hits / max(total, 1)

    print("\n=== verdict ===")
    ok = True

    print(f"1. text<->image cosine: mean {ti.mean():+.3f}, spread {ti.std():.3f}")
    if abs(ti.mean()) < 0.02 and ti.std() < 0.02:
        print("   FAIL - text and image are effectively unrelated (pre-projection "
              "features?). Use get_image_features / the projected embedding.")
        ok = False
    else:
        print("   ok - the two towers share a meaningful space")

    print(f"2. frame-to-frame cosine: mean {ff.mean():.3f}")
    if ff.mean() > 0.97:
        print("   WARN - frames are nearly identical; little temporal signal. "
              "Expected on a fixed camera, but the aligner will struggle.")
    else:
        print("   ok - frames vary over time")

    print(f"3. GT action beats a random action: {pairwise:.2%} (chance = 50%)")
    if pairwise < 0.55:
        print("   FAIL - the semantic term carries almost no information. "
              "Switch to caption-based semantics before extracting everything.")
        ok = False
    else:
        print("   ok - the semantic term is informative")

    print("\n" + ("PASS - safe to run the full extraction"
                  if ok else "DO NOT EXTRACT YET - fix the above first"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
