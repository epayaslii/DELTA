"""Segmentation-timeline plotting.

Render one or more per-frame labelings (ground truth, alignment pseudo-labels,
predicted future, ...) as time-aligned colored tracks with a shared color map.
This is the workhorse for *seeing* where an alignment goes wrong -- used by the
Stage 1 dataset notebook and by Stage 2/3.

No video needed: these operate purely on integer label arrays.
"""

from __future__ import annotations

import numpy as np

_CMAP_NAME = "tab20"


def class_colors(n_classes: int) -> np.ndarray:
    """(n_classes, 3) float RGB in [0,1], stable across calls. tab20 cycled,
    with a fixed light-grey for index 0 (usually a background/START token)."""
    import matplotlib as mpl

    base = mpl.colormaps[_CMAP_NAME].colors  # 20 colors
    cols = np.array([base[i % len(base)][:3] for i in range(max(n_classes, 1))], dtype=float)
    if n_classes > 0:
        cols[0] = (0.85, 0.85, 0.85)
    return cols


def label_strip(labels: np.ndarray, height: int = 24, colors: np.ndarray | None = None) -> np.ndarray:
    """(height, T, 3) uint8 image; column t colored by ``labels[t]``."""
    labels = np.asarray(labels, dtype=int)
    T = len(labels)
    n = int(labels.max()) + 1 if T else 1
    colors = class_colors(n) if colors is None else colors
    row = (colors[np.clip(labels, 0, len(colors) - 1)] * 255).astype(np.uint8)  # (T,3)
    return np.repeat(row[None, :, :], height, axis=0)


def plot_segmentation(
    tracks: dict[str, np.ndarray],
    class_names: list[str] | None = None,
    fps: float | None = None,
    title: str | None = None,
    ax=None,
    boundaries: dict[str, list[int]] | None = None,
    figsize: tuple[float, float] = (12.0, None),  # type: ignore[assignment]
):
    """Plot named label tracks stacked vertically, time-aligned.

    tracks      : {"GT": y_gt, "naive": y_naive, ...} -- each a per-frame int array
    class_names : mapping.txt order; used for the legend
    boundaries  : optional {track_name: [frame_idx, ...]} to mark with vertical ticks
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = list(tracks)
    T = max(len(v) for v in tracks.values())
    n_classes = max(int(np.asarray(v).max()) + 1 for v in tracks.values())
    if class_names:
        n_classes = max(n_classes, len(class_names))
    colors = class_colors(n_classes)

    if ax is None:
        h = figsize[1] or (0.7 * len(names) + 1.6)
        fig, ax = plt.subplots(figsize=(figsize[0], h))
    else:
        fig = ax.figure

    strip_h = 24
    for i, name in enumerate(names):
        y0 = (len(names) - 1 - i) * (strip_h + 6)
        img = label_strip(tracks[name], height=strip_h, colors=colors)
        ax.imshow(img, extent=(0, T, y0, y0 + strip_h), aspect="auto", interpolation="nearest")
        ax.text(-0.01 * T, y0 + strip_h / 2, name, ha="right", va="center", fontsize=10)
        if boundaries and name in boundaries:
            for b in boundaries[name]:
                ax.plot([b, b], [y0, y0 + strip_h], color="k", lw=1.2)

    ax.set_xlim(0, T)
    ax.set_ylim(-4, len(names) * (strip_h + 6))
    ax.set_yticks([])
    if fps:
        ax.set_xlabel("seconds")
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{t / fps:.0f}" for t in ticks])
    else:
        ax.set_xlabel("frame")
    if title:
        ax.set_title(title, fontsize=11)

    if class_names:
        present = sorted(set(np.concatenate([np.unique(v) for v in tracks.values()]).tolist()))
        handles = [
            Patch(facecolor=colors[c], edgecolor="none",
                  label=class_names[c] if c < len(class_names) else str(c))
            for c in present
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax


def segments_text(labels: np.ndarray, class_names: list[str], fps: float | None = None) -> str:
    """Quick textual run-length view of a labeling."""
    from delta.align import segments

    lines = []
    for cls, s, e in segments(labels):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        if fps:
            lines.append(f"{s/fps:7.1f}-{e/fps:7.1f}s  {name}")
        else:
            lines.append(f"{s:6d}-{e:6d}  ({e - s:5d})  {name}")
    return "\n".join(lines)
