# M1 — reproduce MASRA on TACoS

Goal: verify our ESTA/LRCA port (`delta.align.masra_torch`) on a real
grounding pipeline before touching 50Salads. No raw video needed. See
`docs/masra-analysis.md`.

MASRA has **no public code** → we reimplement its two training-time regularizers
on top of **CG-DETR** (`third_party/cgdetr/`, the closest public baseline with a
TACoS config).

## Ready already (local)

- `third_party/cgdetr/` — CG-DETR source (needs Python 3.7 + torch; cluster).
- `third_party/features/tacos/` — SF+C features (305 MB, from the CG-DETR repo):
  ```
  slowfast_features/<vid>.npz   -> features (T, 2304)   # 2 s clips
  clip_features/<vid>.npz       -> features (T, 512)
  clip_text_features/<vid>_<q>.npz -> last_hidden_state (L, 512)
  meta/{train,test,val}.jsonl
  ```
  127 videos, 18227 query features.
- `third_party/cgdetr/data/tacos/{train,test,val}.jsonl` — annotations (in repo).
- `src/delta/align/masra_torch.py` — `esta_loss`, `lrca_loss`, `MasraRegularizer`
  (+ `tests/test_masra_torch.py`, runs on the cluster).

## On the cluster

### 1. Environment
```
conda create -n cgdetr python=3.7 -y && conda activate cgdetr
cd third_party/cgdetr && pip install -r requirements.txt
# feat_root layout: CG-DETR expects ../features/<dset> relative to CGDETR/
ln -s $(pwd)/../features features   # or edit feat_root in the script
```

### 2. M1a — CG-DETR TACoS baseline (no changes)
```
bash cg_detr/scripts/tacos/train.sh
```
Target (CG-DETR paper, TACoS): R1@0.3 ~52.2 / R1@0.5 ~39.6 / R1@0.7 ~22.2 /
mIoU ~36.5. Match within ~1 pt → the pipeline is sound. Results in
`results_tacos/`.

### 3. M1b — add ESTA + LRCA (GT-span targets, no MLLM yet)

Patch points in `third_party/cgdetr/cg_detr/`:

| what | where |
|---|---|
| expose the video memory `E` | `model.py` ~L278 `vid_mem = memory[:, :src_vid.shape[1]]` — add `out["vid_mem"] = vid_mem` |
| per-frame entry index `y` from GT | in `start_end_dataset.py` build a `(T,)` long tensor: frame → index of the GT moment it falls in (TACoS has one span/query → 0 inside, ignore/-1 outside; treat "outside" as its own bucket) |
| event embedding `O` | CLIP-text embedding of the query sentence (already loaded as `src_txt`); pool `last_hidden_state` → 1 row. MASRA's GT-only ESTA variant (Fig. 6). |
| register the loss | `model.py` `SetCriterion.get_loss` `loss_map` — add `"masra": self.loss_masra`; in `loss_masra` call `MasraRegularizer(...)(temporal_ctx=vid_mem, temporal_feat=vid_mem, entry_of_frame=y, transcript=<one entry>, event_emb=O, class_emb=None, relation_mode="block")` |
| weight | `model.py` ~L1025 `weight_dict` — `weight_dict["masra"] = 1.0`; add `"masra"` to `losses` list ~L1044 |

Run: `bash cg_detr/scripts/tacos/train.sh --masra 1` (thread an arg through
`config.py`). Gate: **ESTA+LRCA does not hurt mIoU, and boundary metrics
(R1@0.7) move up** — the direction MASRA reports (Tab. 4).

### 4. M1c — full LRCA with MLLM clip captions  *(optional, needs raw TACoS video)*

Raw TACoS video = MPII Cooking 2 (`mpi-inf.mpg.de/.../mpii-cooking-2-dataset`,
host live). Then: sample clips at the SF+C rate → GPT-5 caption per clip →
CLIP-text encode → cache `caption_features/<vid>.npz (T, 512)` → build `R` from
those (MASRA's `T-T`, its best setting) and pass `relation_mode` accordingly.
This is the same pipeline we reuse for 50Salads at M3.

## Done when

`results_tacos/` baseline matches CG-DETR's paper, and adding `MasraRegularizer`
trains cleanly and nudges the boundary metric the way MASRA's ablation predicts.
Then M2: InternVideo2 features for 50Salads (blocked on raw 50Salads video).
