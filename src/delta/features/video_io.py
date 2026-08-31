"""Deterministic frame sampling.

The extractor must produce one feature vector per *label frame* so the output is
a drop-in replacement for the benchmark I3D features. Given a decoded video with
``n_src`` frames at ``source_fps`` and a target grid of ``n_target`` frames at
``label_fps``, :func:`sample_frame_indices` returns, for every target position,
the source-frame index to decode (nearest-neighbour in time).

For clip / window backbones, :class:`FrameSampler` also yields a small temporal
window of source indices centred on each target position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional at import time so `--help` works without decord installed
    import decord  # type: ignore

    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except Exception:  # pragma: no cover
    _HAS_DECORD = False


def sample_frame_indices(n_src: int, n_target: int, source_fps: float, label_fps: float) -> np.ndarray:
    """Map each of ``n_target`` label positions to a source frame index.

    We place label position ``i`` at time ``(i + 0.5) / label_fps`` seconds and
    pick the closest source frame. Robust to small fps mismatches and to videos
    that are a few frames shorter/longer than the annotation implies.
    """
    if n_target <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_src <= 0:
        raise ValueError("video has no frames")
    t_center = (np.arange(n_target) + 0.5) / float(label_fps)          # seconds
    src_idx = np.round(t_center * float(source_fps)).astype(np.int64)
    return np.clip(src_idx, 0, n_src - 1)


@dataclass
class FrameSampler:
    source_fps: float
    label_fps: float
    window: int = 1          # frames per target position (1 = single frame)
    window_stride: int = 1   # spacing (in source frames) between window samples

    def plan(self, n_src: int, n_target: int) -> np.ndarray:
        """Return an (n_target, window) int array of source indices to decode."""
        centers = sample_frame_indices(n_src, n_target, self.source_fps, self.label_fps)
        if self.window == 1:
            return centers[:, None]
        offs = (np.arange(self.window) - self.window // 2) * self.window_stride
        idx = centers[:, None] + offs[None, :]
        return np.clip(idx, 0, n_src - 1)


class VideoReader:
    """Thin wrapper over decord with a PyAV fallback for whole-video iteration."""

    def __init__(self, path: str, num_threads: int = 2):
        self.path = str(path)
        self._backend = None
        if _HAS_DECORD:
            self._vr = decord.VideoReader(self.path, num_threads=num_threads)
            self._backend = "decord"
            self._n = len(self._vr)
            self._fps = float(self._vr.get_avg_fps())
        else:  # pragma: no cover
            import av

            self._container = av.open(self.path)
            self._stream = self._container.streams.video[0]
            self._backend = "pyav"
            self._fps = float(self._stream.average_rate)
            self._n = self._stream.frames or 0
            if self._n == 0:
                self._n = sum(1 for _ in self._container.decode(video=0))
                self._container.seek(0)

    @property
    def num_frames(self) -> int:
        return self._n

    @property
    def fps(self) -> float:
        return self._fps

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        """Return uint8 RGB frames (N, H, W, 3) for the given source indices."""
        indices = np.asarray(indices, dtype=np.int64)
        if self._backend == "decord":
            return self._vr.get_batch(indices.tolist()).asnumpy()
        # pyav fallback: decode sequentially, keep wanted frames
        want = sorted(set(indices.tolist()))
        got: dict[int, np.ndarray] = {}
        self._container.seek(0)
        for i, frame in enumerate(self._container.decode(video=0)):
            if i in want:
                got[i] = frame.to_ndarray(format="rgb24")
                if len(got) == len(want):
                    break
        return np.stack([got[i] for i in indices], axis=0)

    def close(self) -> None:
        if self._backend == "pyav":  # pragma: no cover
            self._container.close()
