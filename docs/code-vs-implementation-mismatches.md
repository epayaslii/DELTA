# Open questions for the group's engineer — code vs. our implementation

Compiled from working directly with `third_party/delta_wlta` (gitignored,
shared snapshot) against the DELTA paper + supplementary. Ranked by how much
each one blocks a trustworthy comparison.

## 1. Blocking — the official 12-class eval mapping doesn't exist in our copy

Table 1's headline anticipation number (20.92 deterministic MoC on 50Salads)
is computed at **`-d FSeval`, `-c 12`** — a coarser grouping than the 19-class
mid granularity (`-d FS`, `-c 19`) our data actually is. The loader expects a
`mappingeval.txt` we don't have and haven't been able to derive (the exact
12-class grouping — which mid classes merge into which eval classes — isn't
in the paper or supplementary). **Without this, we cannot produce a number
directly comparable to Table 1**, only to the 19-class TAS/segmentation stage.
→ **Ask for `mappingeval.txt`, or the exact 12-class grouping rule.**

## 2. Blocking-ish — exact I3D feature provenance

Confirmed: the Kaggle `asad1212/50salads` and HF `dinggd/50salads` bundles are
byte-identical (checked all 50 videos — features, groundTruth, mapping,
splits). Not confirmed: whether **either** matches the exact features used to
produce the paper's reported numbers, vs. a different extraction/version.
→ **Ask directly**: is `dinggd/50salads` (or the identical Kaggle copy) the
same I3D features the manuscript's 50Salads results were computed on?

## 3. Correctness issue in the shipped code — worth flagging, not just asking

`train.py::on_test_epoch_end` scores segmentation quality (`50S_final_metrics.json`'s
`MOF (%)` / `mIOU (%)`) via `ClusteringMetrics` → `pred_to_gt_match` →
`scipy.optimize.linear_sum_assignment` — Hungarian bipartite label matching.
That's correct for unsupervised clustering (arbitrary cluster IDs) but **wrong
here**, since the TAS classifier's predictions already carry true semantic
class IDs — remapping them before scoring inflates the number. Confirmed
independently on a from-scratch baseline (see `50salads-notes.md`'s resolved
discrepancy note): with the fix, that baseline went from 0.582 → 0.380.
The **anticipation MoC** (`calculate_moc`, feeding Table 1) is *not* affected
— `eval_file` uses a literal identity class mapping. Only the segmentation-
stage MOF/mIoU is contaminated.
→ **Flag this** — anyone reading `50S_final_metrics.json`'s MOF/mIoU at face
value (rather than the anticipation grid) is getting an inflated number.
We're bypassing it (`scripts/dump_ta_pseudolabels.md` + `score_ta_dump.py`),
but worth knowing it's there for future use of the code.

## 4. Documentation vs. code — the stage transition

Supplementary text: the stage-1→stage-2 transition happens "until the
segmentation head yields temporally consistent pseudo-labels" — described as
criterion-based. The actual code (`train.py:288`) hardcodes
`stage1to2 = 10, stage2to3 = 30` (fixed epochs, not a criterion check).
→ **Ask**: is the hardcoded 10/30 schedule what produced the paper's numbers,
or was a criterion-based version used and only the fallback/default left in
the shared snapshot?

## 5. Missing file

Every `run_*.sh` driver script calls `src/train_edit_elena9sept.py`, which
isn't in the snapshot — only `train.py` and `train_window_tokenizer.py` exist.
The scripts' flags map cleanly onto `train.py`, and that's what we've been
using.
→ **Confirm** `train.py` is the right substitute, or ask for the missing file
if it has logic not present in `train.py`.

## 6. Environment / portability (low stakes, just needs doing)

- Hardcoded absolute paths (`/home/ebueno/datasets/...`, `/home/datasets/clot_transcripts/...`) — need `--base-path`/`--root` overrides to run anywhere else, including GCP.
- `README.md` in the snapshot is CLOT's, not DELTA's (conda env named `clot`).
- `requirements_env.txt` is a full conda export — Python 3.9 / torch 1.11 / CUDA 11.3 specifically.

## What's already resolved (no need to re-ask)

- Full hyperparameter table (batch/epochs/lr/decoder layers/loss weights) — from the supplementary, in `delta-code.md`.
- The two-vs-three granularity confusion (19 mid / 17 action-only / 12 eval) — worked out from the code, see `delta-code.md`.
- Data-source discrepancy with the parallel workstream — resolved, was a metric bug (item 3 above), not the data.
