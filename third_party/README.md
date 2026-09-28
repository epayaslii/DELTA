# third_party/

External code, not committed here (see `.gitignore`).

## delta_wlta/ — the DELTA implementation ("WLTA")

Shared by the professor's group (snapshot Oct 2025). **This is what we build
on.** Full analysis + how the experiments run: [`../docs/delta-code.md`](../docs/delta-code.md).
TL;DR: DELTA on top of the CLOT/ASOT codebase; `--model_type atba` uses the
ATBA boundary detector (`src/atba_loss.py`), `wclot` uses ASOT optimal transport
(`src/asot.py`); 50Salads driver is `run_50S_allmetrics.sh`; needs
Linux+CUDA+conda+wandb.

## atba/ — ATBA (Xu & Zheng, CVPR 2024)

```bash
git clone https://github.com/iSEE-Laboratory/CVPR24_ATBA.git third_party/atba
```
Standalone ATBA. Mostly superseded by the copy already inside `delta_wlta`
(`src/atba_loss.py`); keep for cross-checking the DP against the paper's Fig. 6.

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

## cgdetr/ — CG-DETR (Moon et al., CVPR 2024)

```bash
git clone https://github.com/wjun0830/CGDETR.git third_party/cgdetr
```
Video-temporal-grounding DETR; the public baseline closest to **MASRA** (which
has no code). We reimplement MASRA's ESTA/LRCA regularizers on top of it for the
M1 TACoS check — see [`../scripts/masra_m1.md`](../scripts/masra_m1.md),
[`../docs/masra-analysis.md`](../docs/masra-analysis.md). Needs Python 3.7.

## features/ — pre-extracted grounding features (gitignored)

`features/tacos/` — CG-DETR's SlowFast+CLIP features for TACoS (305 MB, from the
CG-DETR repo's Google Drive). `slowfast_features` (T, 2304), `clip_features`
(T, 512), `clip_text_features` per query (L, 512). 127 videos.


## delta_wlta_supervisor_copy/ — fresh copy from the supervisor (gitignored)

Sent 2026-09-22 alongside the D-CLOT code, to help debug the 17.15 vs 19.11 MoC
reproduction gap. Confirmed byte-identical to `delta_wlta/` for
`train_window_tokenizer.py` / `atba_loss.py` (our patch targets are the real
bugs, not something already fixed upstream). Extra files not in our copy:
`configs_wandb/{hyperparameter,sweep}.yaml` and `src/sweep_LTA.yaml` — these are
hyperparameter-sweep *search grids* (e.g. `LTA_dec_layers: [2, 3, 4]`), not the
final chosen values; they confirm 3 was a legitimate candidate but don't pin
down which one produced the published number. No checkpoints included.

## d_clot/ — D-CLOT / CLOT++ (gitignored)

The supervisor's follow-up to CLOT, sent as code only (the mentioned weights
were not actually included — ask for them again if needed). Covers 50Salads via
`run_50s_CLOT++_sota.sh` / `run_50s_CLOT++B_sota.sh`, and is the only place so
far referencing `mapping/mappingeval.txt` (12-class eval granularity, Table 1's
20.92) — the file itself still isn't bundled. Relevant if we revisit optional
Stage C (closed-loop frame/segment OT refinement).
