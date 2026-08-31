# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stage 1 — 50Salads dataset analysis
#
# Goal: understand *why* 50Salads is hard for transcript-only temporal alignment,
# using only the benchmark bundle (`groundTruth/` + `mapping.txt` + I3D `features/`).
# No raw video (the official host is down).
#
# Everything here uses `delta.data.stats`, `delta.align`, `delta.viz`.

# %%
import os
from pathlib import Path

# run from the repo root regardless of where the notebook is launched
_p = Path.cwd()
while _p != _p.parent and not (_p / "pyproject.toml").exists():
    _p = _p.parent
os.chdir(_p)
print("cwd:", os.getcwd())

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from delta.data import ActionSegDataset
from delta.data import stats
from delta.align import segmentation_report, segments
from delta.viz import plot_segmentation

pd.set_option("display.max_rows", 60)
ds = ActionSegDataset("50salads", "data/50salads")
FPS = stats.label_fps(ds)
NAMES = ds.action_names()
ds, FPS, len(NAMES)

# %% [markdown]
# ## 1. Basic facts

# %%
vdf = stats.video_df(ds)
print(f"{len(vdf)} videos, {len(NAMES)} classes, {FPS:.0f} fps")
print(f"split1 train/test: {len(ds.split(1,'train'))}/{len(ds.split(1,'test'))}")
vdf.describe().loc[["mean", "min", "max"]].T

# %% [markdown]
# ~50 videos, ~20 action segments each, ~6.4 min long, ~30 fps → **long sequences
# with many fine-grained transitions**. `action_start`/`action_end` wrap every
# video (~14% of frames).

# %% [markdown]
# ## 2. Class frequency — is it long-tailed?

# %%
fc = stats.frame_counts(ds)
sc = stats.segment_counts(ds)
freq = pd.DataFrame({
    "class": [NAMES[i] for i in range(len(NAMES))],
    "frames": [fc.get(i, 0) for i in range(len(NAMES))],
    "segments": [sc.get(i, 0) for i in range(len(NAMES))],
})
freq["frame_share_%"] = 100 * freq["frames"] / freq["frames"].sum()
freq = freq.sort_values("frames", ascending=False).reset_index(drop=True)
freq

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(freq["class"], freq["frame_share_%"]); ax[0].set_title("frame share (%)")
ax[0].tick_params(axis="x", rotation=90)
ax[1].bar(freq["class"], freq["segments"]); ax[1].set_title("# segments")
ax[1].tick_params(axis="x", rotation=90)
fig.tight_layout()

# %% [markdown]
# The `cut_* / place_*_into_bowl` families and `mix_*` dominate frame count, while
# the `add_{salt,vinegar,oil,pepper}` actions are short and rare in frames but
# each appears once per recipe. A **frame-weighted** loss (what fully-supervised
# methods optimise) is dominated by the long actions; a **transcript**-weighted
# view counts every action once — the paper's "frequency bias vs procedural
# structure" point. → **MoC (per-class), not MoF, is the headline metric.**

# %% [markdown]
# ## 3. Segment-duration variability per class
#
# Coefficient of variation (`cv = std/mean`): the most variable classes sit at
# **cv ≈ 0.5–0.8** (e.g. `place_tomato_into_bowl`: mean ~8 s, cv 0.78 → lengths
# roughly 2–20 s). A single ordered transcript occurrence tells you little about
# extent ⇒ the duration head has weak signal (paper: duration loss = **+0.2 MoC
# on 50S** vs +3.3 on Breakfast).

# %%
cdf = stats.class_duration_df(ds).sort_values("cv", ascending=False).reset_index(drop=True)
cdf.round(2)

# %%
sdf = stats.segment_df(ds)
order = cdf.sort_values("mean_sec")["class_name"].tolist()
fig, ax = plt.subplots(figsize=(13, 4))
data = [sdf.loc[sdf["class_name"] == c, "dur_sec"].values for c in order]
ax.boxplot(data, showfliers=False)
ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=90)
ax.set_ylabel("segment duration (s)")
ax.set_title("per-class segment-duration spread"); fig.tight_layout()

# %% [markdown]
# ## 4. How much does transcript *order* pin down the next action?
#
# `transition_matrix[c]` = P(next action | current = c) over all transcripts.
# Per-row entropy: 0 = deterministic successor, higher = ambiguous order.

# %%
P = stats.transition_matrix(ds, normalize=True)
ent = stats.transition_entropy(ds)
real = [i for i in range(len(NAMES)) if NAMES[i] not in ("action_start", "action_end")]

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(P[np.ix_(real, real)], cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(real))); ax.set_xticklabels([NAMES[i] for i in real], rotation=90, fontsize=7)
ax.set_yticks(range(len(real))); ax.set_yticklabels([NAMES[i] for i in real], fontsize=7)
ax.set_title("P(next | current)"); fig.colorbar(im, fraction=0.046); fig.tight_layout()

# %%
mean_ent = np.nanmean([ent[i] for i in real])
print(f"mean next-action entropy over real classes: {mean_ent:.2f} bits "
      f"(~{2**mean_ent:.1f} effective successors)")
pd.Series({NAMES[i]: ent[i] for i in real}).sort_values(ascending=False).round(2)

# %% [markdown]
# ~1.9 bits ⇒ each action has ~3-4 plausible successors. **The transcript order
# alone does not disambiguate the timeline** — the visual transition score `V^a`
# has to do real work, and on 50Salads the classes it must separate
# (`cut_tomato`/`cut_cheese`/`cut_lettuce`/`cut_cucumber`,
# `add_oil`/`add_vinegar`/`add_salt`/`add_pepper`) look nearly identical from a
# fixed overhead camera.

# %% [markdown]
# ## 5. Do the I3D features even show the boundaries?
#
# ATBA's candidate boundaries come from a *class-agnostic* score built on frame
# feature change. Proxy: cosine distance between consecutive I3D frames, overlaid
# on the ground-truth boundaries. If boundaries don't stand out here, the
# candidate set will miss true transitions no matter how good `V^a` is.

# %%
def boundary_proxy(vid, smooth=15):
    f = np.load(f"data/50salads/features/{vid}.npy").astype(np.float32)  # (D,T)
    f = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-8)
    d = 1.0 - (f[:, 1:] * f[:, :-1]).sum(0)                              # (T-1,)
    k = np.ones(smooth) / smooth
    d = np.convolve(d, k, mode="same")
    y = ds.record(vid).frame_labels
    bnds = [s for _, s, _ in segments(y)][1:]
    return d, bnds, y

vid = ds.split(1, "test")[0]
d, bnds, y = boundary_proxy(vid)
fig, ax = plt.subplots(2, 1, figsize=(13, 4), sharex=True, height_ratios=[3, 1])
ax[0].plot(d, lw=0.7); [ax[0].axvline(b, color="r", lw=0.8, alpha=0.7) for b in bnds]
ax[0].set_title(f"{vid}: consecutive-frame I3D cosine distance (red = GT boundaries)")
ax[1].imshow(y[None, :], aspect="auto", cmap="tab20"); ax[1].set_yticks([])
fig.tight_layout()

# %%
# quantify: is the proxy higher at GT boundaries than at random frames?
rng = np.random.default_rng(0)
at_b, at_r = [], []
for vid in ds.split(1, "test"):
    d, bnds, y = boundary_proxy(vid)
    bnds = [b for b in bnds if 0 < b < len(d)]
    at_b += [d[b] for b in bnds]
    at_r += list(d[rng.integers(0, len(d), size=len(bnds))])
at_b, at_r = np.array(at_b), np.array(at_r)
print(f"I3D cosine-distance at GT boundaries : {at_b.mean():.4f}")
print(f"                  at random frames  : {at_r.mean():.4f}")
print(f"ratio: {at_b.mean()/at_r.mean():.2f}x  (closer to 1.0 = boundaries barely stand out)")

# %% [markdown]
# ## 6. The floor: naive uniform alignment
#
# Split each video into `len(transcript)` equal parts, in order — the pseudo-label
# you get with **zero** visual evidence. Any real TA method must beat this.

# %%
rows = []
for vid in ds.all_ids():
    rec = ds.record(vid)
    y_naive = stats.naive_uniform_labeling(rec.transcript, rec.num_label_frames)
    r = segmentation_report(y_naive, rec.frame_labels)
    r["video_id"] = vid
    rows.append(r)
naive = pd.DataFrame(rows).set_index("video_id")
naive.mean(numeric_only=True).round(3)

# %% [markdown]
# ## 7. See it: GT vs naive alignment

# %%
for vid in ds.split(1, "test")[:2]:
    rec = ds.record(vid)
    y_naive = stats.naive_uniform_labeling(rec.transcript, rec.num_label_frames)
    fig, _ = plot_segmentation(
        {"GT": rec.frame_labels, "naive uniform": y_naive},
        class_names=NAMES, fps=FPS,
        title=f"{vid}  —  naive MoF={segmentation_report(y_naive, rec.frame_labels)['MoF']:.2f}",
    )
    plt.show()

# %% [markdown]
# ## 8. Breakfast comparison (deferred)
#
# Needs the Breakfast bundle (`dinggd/breakfast`, ~larger). Once downloaded:
# ```python
# bf = ActionSegDataset("breakfast", "data/breakfast")
# pd.DataFrame([stats.dataset_summary(ds), stats.dataset_summary(bf)])
# ```
# Expected contrast: BF ~6 segments/video vs 50S ~20; fewer, more separable steps
# per video → the alignment "locks on" quickly on BF but not on 50S.

# %% [markdown]
# ## Takeaways → `docs/50salads-notes.md`
#
# 1. **MoC over MoF** — frame counts are dominated by a few long actions; the
#    interesting classes (`add_*`) are frame-rare but appear once per recipe.
# 2. **Duration is hard to learn from transcripts** — variable classes at CV ≈ 0.5–0.8.
# 3. **Order ≠ timing** — ~1.9 bits of next-action entropy; visual evidence must
#    carry the alignment, and…
# 4. **…the visual evidence is weak** — fixed overhead camera, near-duplicate
#    fine-grained classes, I3D consecutive-frame distance barely peaks at true
#    boundaries (§5 ratio). This is the ATBA `v^b` candidate-generation problem.
# 5. **Long sequences** (~20 segments, ~11k frames) → one early boundary error
#    propagates far.
