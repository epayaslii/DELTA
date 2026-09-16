"""Score a raw TA/segmentation dump from the DELTA/WLTA code with our clean,
direct-label metrics -- bypassing its own Hungarian-matched MOF/mIoU.

See scripts/dump_ta_pseudolabels.md for how the dump is produced and why this
exists. No torch required; runs anywhere with this repo's .venv.

Usage:
    python scripts/score_ta_dump.py results/50S/ta_dump_atba_s1.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from delta.align import segmentation_report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump", help=".npz written by the on_test_epoch_end patch")
    p.add_argument("--ignore-startend", action="store_true",
                   help="exclude action_start/action_end classes (17-class; "
                        "diagnostic only -- 19-class mid granularity is our default, see 50salads-notes.md)")
    args = p.parse_args(argv)

    d = np.load(args.dump, allow_pickle=True)
    preds, gts = d["preds"], d["gts"]
    fnames = d["fnames"] if "fnames" in d else [None] * len(preds)
    ignore = {17, 18} if args.ignore_startend else set()

    rows = []
    for pred, gt, fname in zip(preds, gts, fnames):
        pred, gt = np.asarray(pred), np.asarray(gt)
        if len(pred) == 0 or len(gt) == 0:
            continue
        r = segmentation_report(pred, gt, ignore=ignore)
        r["fname"] = str(fname)
        rows.append(r)

    if not rows:
        print("no scoreable entries in the dump")
        return 1

    keys = [k for k in rows[0] if k != "fname"]
    mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}

    print(f"[score_ta_dump] {args.dump}  (n={len(rows)} video/obs-ratio entries, "
         f"{'17' if args.ignore_startend else '19'}-class)")
    print(json.dumps({k: round(v, 4) for k, v in mean.items()}, indent=2))
    print("\nNote: this replaces the run's own printed MOF (%) / mIOU (%), which "
         "are Hungarian-matched and inflated -- see docs/delta-code.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
