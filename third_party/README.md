# third_party/

External code used as **baselines**, not committed here (see `.gitignore`).
Clone locally:

```bash
git clone https://github.com/iSEE-Laboratory/CVPR24_ATBA.git third_party/atba
```

## atba/ — ATBA (Xu & Zheng, CVPR 2024)

"Alignment through segmentation" transcript-supervised baseline. DELTA's TA
module follows its boundary detector. Official PyTorch impl (Python 3.9 +
PyTorch 1.11, single GPU). Reports Breakfast / Hollywood / CrossTask — **not
50Salads**; we add a 50Salads config and run it on the cluster for the `Y*`
comparison table (`docs/temporal-alignment.md` §5.4).

Its benchmark data (groundTruth/splits) is on the authors' Google Drive; the
50Salads I3D features we already have under `data/50salads/` (from
`dinggd/50salads`) are compatible.

## HAL (optional second baseline)

`arXiv:2602.24275` (CVPR 2026) — current transcript-supervised SOTA. Also does
not report 50Salads.
