"""Foundation-model feature extraction for 50Salads / Breakfast.

Emits one ``<video_id>.npy`` of shape ``(D, T)`` per video, where ``T`` matches
the length of the corresponding ``groundTruth/<video_id>.txt`` -- a drop-in
replacement for the benchmark I3D features.

Examples
--------
# all 50Salads videos, VideoLLaMA3 vision tower, on GPU 0
python -m delta.features.extract --config configs/50salads.yaml

# only test split 1, and resume (skip finished)
python -m delta.features.extract --config configs/50salads.yaml --split 1 --subset test

# cluster array job: 8 shards
python -m delta.features.extract --config configs/50salads.yaml --shard $SLURM_ARRAY_TASK_ID/8

# dump action-name text embeddings (once)
python -m delta.features.extract --config configs/50salads.yaml --text-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from delta.data import ActionSegDataset
from delta.data.datasets import DATASET_DEFAULTS
from delta.features.video_io import FrameSampler, VideoReader


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def target_length(rec, reader: VideoReader, label_fps: float) -> int:
    if rec.num_label_frames:
        return rec.num_label_frames
    # no ground truth (e.g. inference-only video): derive from duration
    dur_s = reader.num_frames / max(reader.fps, 1e-6)
    return max(1, int(round(dur_s * label_fps)))


def extract_one(rec, backbone, sampler: FrameSampler, label_fps: float, l2norm: bool) -> np.ndarray:
    reader = VideoReader(str(rec.video_path))
    try:
        T = target_length(rec, reader, label_fps)
        plan = sampler.plan(reader.num_frames, T)          # (T, window)
        flat = plan.reshape(-1)
        frames = reader.get_batch(flat)                    # (T*window, H, W, 3)
        feats = backbone.encode(frames)                    # (T*window, D)
    finally:
        reader.close()
    feats = feats.reshape(plan.shape[0], plan.shape[1], -1).mean(axis=1)   # window pool -> (T, D)
    if l2norm:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats.astype(np.float32).T                      # (D, T)  -- I3D layout


def select_ids(ds: ActionSegDataset, args) -> list[str]:
    if args.ids:
        return args.ids
    if args.split is not None:
        return ds.split(args.split, args.subset)
    return ds.all_ids()


def parse_shard(s: str | None) -> tuple[int, int]:
    if not s:
        return 0, 1
    i, n = s.split("/")
    return int(i), int(n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--backbone", help="override config.features.backbone")
    p.add_argument("--out-dir", help="override output dir")
    p.add_argument("--split", type=int, help="restrict to a benchmark split")
    p.add_argument("--subset", default="test", choices=["train", "test"])
    p.add_argument("--ids", nargs="*", help="explicit video ids")
    p.add_argument("--shard", help="'i/n' for cluster array jobs (0-indexed)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--text-only", action="store_true", help="only dump action-name embeddings")
    p.add_argument("--limit", type=int, help="debug: process at most N videos")
    args = p.parse_args(argv)

    from delta.features.backbones import build_backbone  # noqa: local import (heavy deps)
    from delta.features.text_encoder import encode_action_names

    cfg = load_config(args.config)
    dname = cfg["dataset"]["name"].lower()
    defaults = DATASET_DEFAULTS[dname]
    feat_cfg = cfg.get("features", {})
    backbone_name = args.backbone or feat_cfg.get("backbone", "vl3-siglip")

    ds = ActionSegDataset(
        name=dname,
        root=cfg["dataset"]["root"],
        video_dir=cfg["dataset"].get("video_dir"),
        feature_dir=cfg["dataset"].get("feature_dir"),
    )

    out_dir = Path(args.out_dir or feat_cfg.get("out_dir") or (ds.root / f"features_{backbone_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- text embeddings ----------------------------------------------------------
    text_enc = feat_cfg.get("text_encoder", "siglip2")
    text_path = out_dir / "action_name_embeddings.npy"
    if args.text_only or not text_path.exists():
        names = ds.action_names()
        if names:
            emb = encode_action_names(names, encoder=text_enc, device=args.device,
                                      raw=feat_cfg.get("text_raw", False))
            np.save(text_path, emb)
            (out_dir / "action_names.json").write_text(json.dumps(names, indent=2))
            print(f"[text] {emb.shape} -> {text_path}")
        else:
            print("[text] mapping.txt not found; skipping action-name embeddings")
    if args.text_only:
        return

    # ---- video features ---------------------------------------------------------
    backbone = build_backbone(
        backbone_name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
    )
    sampler = FrameSampler(
        source_fps=feat_cfg.get("source_fps", defaults["source_fps"]),
        label_fps=feat_cfg.get("label_fps", defaults["label_fps"]),
        window=feat_cfg.get("window", 1),
        window_stride=feat_cfg.get("window_stride", 1),
    )
    l2norm = feat_cfg.get("l2_normalize", True)

    ids = select_ids(ds, args)
    shard_i, shard_n = parse_shard(args.shard)
    ids = [v for k, v in enumerate(ids) if k % shard_n == shard_i]
    if args.limit:
        ids = ids[: args.limit]

    manifest = out_dir / f"manifest.shard{shard_i}of{shard_n}.jsonl"
    print(f"[video] backbone={backbone_name} dim={backbone.dim} n_videos={len(ids)} -> {out_dir}")

    done = skipped = failed = 0
    with open(manifest, "a") as mf:
        for vid in ids:
            dst = out_dir / f"{vid}.npy"
            if dst.exists() and not args.overwrite:
                skipped += 1
                continue
            rec = ds.record(vid)
            if not rec.has_video():
                print(f"  !! no video file for {vid}")
                failed += 1
                continue
            t0 = time.time()
            try:
                feats = extract_one(rec, backbone, sampler, sampler.label_fps, l2norm)
            except Exception as e:  # keep the array job alive
                print(f"  !! {vid}: {type(e).__name__}: {e}")
                failed += 1
                continue
            np.save(dst, feats)
            done += 1
            rowsrc = {
                "video_id": vid, "shape": list(feats.shape), "dt": round(time.time() - t0, 2),
                "n_label_frames": rec.num_label_frames, "transcript_len": len(rec.transcript or []),
            }
            mf.write(json.dumps(rowsrc) + "\n")
            mf.flush()
            print(f"  ok {vid} {feats.shape} ({rowsrc['dt']}s)")

    print(f"[done] extracted={done} skipped={skipped} failed={failed}")
    if backbone_name != feat_cfg.get("backbone"):
        print(f"note: point DELTA's data loader at {out_dir}")


if __name__ == "__main__":
    main()
