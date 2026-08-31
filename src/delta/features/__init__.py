"""Feature extraction. Heavy deps (torch, transformers) are imported lazily so
that ``video_io`` helpers and configs can be used without them installed.
"""

from .video_io import FrameSampler, sample_frame_indices

__all__ = [
    "FrameSampler",
    "sample_frame_indices",
    "build_backbone",
    "list_backbones",
    "FrameBackbone",
    "build_text_encoder",
    "encode_action_names",
]


def __getattr__(name):  # PEP 562 lazy re-export
    if name in ("build_backbone", "list_backbones", "FrameBackbone"):
        from . import backbones

        return getattr(backbones, name)
    if name in ("build_text_encoder", "encode_action_names"):
        from . import text_encoder

        return getattr(text_encoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
