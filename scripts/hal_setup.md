# Phase T setup — ATBA & HAL on 50Salads

Goal: run the standalone **ATBA** and **HAL** repos on 50Salads, extract `Y*`
on the held-out split, score with `delta.align`. See
`docs/baselines-hal-cva.md` (Phase T).

## Ready already (local, in this repo)

`data/50salads/` (from `dinggd/50salads` + `scripts/make_transcripts.py`):
```
data/50salads/
  features/<vid>.npy      # (2048, T) I3D, float
  groundTruth/<vid>.txt   # one action-name per line
  transcripts/<vid>.txt   # ordered action list (generated — RLE of groundTruth)
  mapping.txt             # "<id> <name>", 19 classes (0..16 actions, 17 action_start, 18 action_end)
  splits/{train,test}.split{1..5}.bundle
```
This is exactly the layout ATBA/HAL's `datasets.py::MyDataset` expects, as
`data/50salads/` under `--root`.

Both repos are cloned to `third_party/{atba,hal}/` (gitignored).

## On the cluster

### 1. Environment (shared by ATBA, HAL, and later WLTA)
```
conda create -n wsas python=3.9.23 -y && conda activate wsas
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy scipy scikit-learn matplotlib tensorboardX
```

### 2. Copy the data
```
rsync -a data/50salads  <cluster>:<path>/ATBA/data/     # -> data/50salads/
rsync -a data/50salads  <cluster>:<path>/HAL/data/
```
(regenerate transcripts on the cluster if needed: `python scripts/make_transcripts.py data/50salads`)

### 3. Patches (3 small edits — the repos have vestigial 50salads support)

**ATBA `datasets.py`:**
- line ~9: `assert dataset_name in ('breakfast', 'hollywood', 'crosstask', '50salads')`
- line ~55: `if self.dataset_name in ('breakfast', 'crosstask', '50salads'):  feat = feat.T`
  *(our `.npy` are stored (2048, T); ATBA only transposes breakfast/crosstask by default → shape-mismatch assert without this)*

**HAL `datasets.py`:**
- line ~11: `assert dataset_name in ('breakfast', 'hollywood', 'crosstask', 'gtea', '50salads')`
  *(HAL already includes `'50salads'` in its transpose list and `model.py` already has a `max_len=20000` branch)*

**HAL `options.py`:** add
```
parser.add_argument('--lags', type=int, default=1)
```
*(`model.py:68` does `self.lags = self.args.lags` but the arg is missing → AttributeError at init. `self.lags` is otherwise unused, so `--lags 1` is a no-op.)*

Note: both repos hardcode `bg_cls = 0`. 50Salads class 0 is `cut_tomato`, not
background — this only affects `MoF-Bg` and the `--bgw` weighting, not MoF / MoC.
Leave it; just don't trust MoF-Bg for 50S.

### 4. Train — split 1 first (minimal-viable gate: 1 split, 1 seed)

**ATBA** (fast — ~1–2 h):
```
cd third_party/atba
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 50salads --root ./data/ \
  --split 1 --sample-rate 10 --seed 0 --epoch 400 --cs-kernel 31 \
  --exp-name atba_50s_s1 --save
```

**HAL** (~2–4 h):
```
cd third_party/hal
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 50salads --root ./data/ \
  --split 1 --sample-rate 10 --seed 0 --epoch 400 --cs-kernel 31 --n-encoder 5 \
  --rec_weight 0.1 --diff_weight 1e-3 --kl_weight 1e-3 --lags 1 \
  --exp-name hal_50s_s1 --save
```
`--sample-rate 10` on 50S (~11.5k frames @ 30 fps) → ~1150 sampled frames;
`pos_embedding` covers `20000/10 = 2000`. Try `--sample-rate 4` if you want
finer resolution.

Both write TensorBoard logs to `./logs/<exp-name>/` and, with `--save`,
`<exp-name>.npy` with `{acc, acc-bg, iou_isba, iod_isba, iou_tasl, iod_tasl}`
from `test.py::test_all(..., fully_eva=True)`.

### 5. Extract `Y*` and score with `delta.align` (Phase T4)

`test.py` already computes MoF (`acc`), IoU, IoD. For the fuller TA metrics
(MoC, Edit, F1@{10,25,50}, boundary offset) add a dump: in `main.py` after the
final `test_all`, or in `test.py::test_all`, collect `pred_lst` / `gt_lst`
(already there) and:
```python
import numpy as np
np.savez("Y_star_atba_s1.npz",
         preds=np.array(pred_lst, dtype=object), gts=np.array(gt_lst, dtype=object))
```
Then, back in this repo:
```python
import numpy as np
from delta.align import segmentation_report
d = np.load("Y_star_atba_s1.npz", allow_pickle=True)
rows = [segmentation_report(p, g) for p, g in zip(d["preds"], d["gts"])]
mean = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
print(mean)   # MoF, MoC, edit, F1@10/25/50
```
Do the same for HAL → the Phase T4 comparison table.

For **per-video visualisation** (understanding, not the metric):
```python
from delta.viz import plot_segmentation
plot_segmentation({"GT": g, "ATBA Y*": p_atba, "HAL Y*": p_hal},
                  class_names=mapping_names, fps=30)
```

### 6. Gate

`Y*_HAL` vs `Y*_ATBA` vs naive floor (MoC 0.34), on MoC + boundary offset.
Beyond seed noise → Phase D (port HAL into DELTA `--model_type atba`).
Not → stop, record the numbers.
