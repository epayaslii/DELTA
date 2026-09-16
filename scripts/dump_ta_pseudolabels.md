# Dumping DELTA's TA/segmentation output for independent (clean) scoring

**Why:** DELTA's own `on_test_epoch_end` scores segmentation quality via
`self.mof`/`self.miou` → `ClusteringMetrics` → Hungarian bipartite label
matching (`pred_to_gt_match` → `scipy.optimize.linear_sum_assignment`) —
correct for unsupervised clustering, **wrong once predictions already carry
true semantic class IDs** (see `delta-code.md`'s "shipped code's own MOF/mIoU"
section — same bug a parallel workstream found in their own baseline). Their
own `50S_final_metrics.json` `MOF (%)` / `mIOU (%)` are inflated by it.

**The fix doesn't need new forward-pass code.** `test_step` already
accumulates exactly what we need, unconditionally, into `self.test_cache`:
```python
self.test_cache.append([metrics['mof'], segments, obs_gt, obs_mask, fname])
```
(`train.py` ~line 1321, both `atba` and `wclot` branches feed the same cache).
`segments` = the trained TAS classifier's argmax over the **observed prefix**
only (no transcript — this is the genuine inference-time output, matching the
paper's "transcript discarded at inference"); `obs_gt` = the matching
ground-truth slice; one entry per `(video, observation_ratio)` pair, for every
`alpha` in `self.observation_ratios` (the same 20%/30% grid as the DLTA eval).
`metrics['mof']` in that tuple is a *different*, non-cached-metric per-item MoF
(via `indep_eval_metrics`, not `ClusteringMetrics` -- check whether that one
also takes a `pred_to_gt` arg before trusting it; the dump below sidesteps the
question entirely by rescoring from raw `segments`/`obs_gt`).

## Patch — `third_party/delta_wlta/src/train.py`

In `on_test_epoch_end`, **before** the existing `self.test_cache = []` reset
(~line 1487) and before the Hungarian-matched `self.mof.compute()` block
(~line 1434) invalidates anything, add:

```python
def on_test_epoch_end(self):
    # --- dump raw predictions for independent scoring (no Hungarian matching) ---
    import numpy as np
    dump_path = getattr(args, "ta_dump_path", None)
    if dump_path and self.test_cache:
        preds, gts, fnames = [], [], []
        for _, segments, obs_gt, obs_mask, fname in self.test_cache:
            m = obs_mask[0].bool().cpu().numpy()
            preds.append(segments[0].detach().cpu().numpy()[m])
            gts.append(obs_gt[0].detach().cpu().numpy()[m])
            fnames.append(fname[0] if isinstance(fname, (list, tuple)) else fname)
        np.savez(dump_path, preds=np.array(preds, dtype=object),
                 gts=np.array(gts, dtype=object), fnames=np.array(fnames))
        print(f"[ta-dump] wrote {len(preds)} (video, obs-ratio) entries -> {dump_path}")
    # --- existing body follows unchanged ---
    mof, pred_to_gt, accumulate_mof = self.mof.compute(exclude_cls=self.exclude_cls)
    ...
```

Add the CLI flag (near the other `argparse` definitions, top of `train.py`):
```python
parser.add_argument('--ta_dump_path', type=str, default=None,
                    help='if set, dump raw (pred, gt) per test video/obs-ratio here, bypassing the Hungarian-matched MOF/mIoU')
```

Run exactly like `scripts/slurm_delta_baselines.sh` already does, adding one flag:
```bash
python train.py --dataset 50salads --split 1 --model_type atba \
  --ta_dump_path results/50S/ta_dump_atba_s1.npz ...
```

## Scoring — no torch needed, runs anywhere

```bash
python scripts/score_ta_dump.py results/50S/ta_dump_atba_s1.npz
```
Loads the `.npz`, scores every entry with `delta.align.segmentation_report`
(direct-label — no remapping, verified clean), reports MoF/MoC/F1@k averaged
per (video, observation-ratio) and overall. **This is the number to compare
against — not the run's own printed `MOF (%)`.**

## What this does *not* give us

`segments` is the **inference-time TAS classifier's output on the observed
prefix**, not the literal train-time `Y*` (the ATBA boundary-detector + DP
output, which needs a transcript and is never computed for held-out video by
the shipped code — see `delta-code.md`). It's a legitimate, meaningful
generalisation-to-unseen-video number and the right thing to compare against
our own pipeline's segmentation quality, but it is not "run TA on a held-out
video with its transcript." That would be a larger, unwritten addition if we
decide we need it specifically.
