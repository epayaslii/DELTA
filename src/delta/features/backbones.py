"""Frozen frame/clip feature backbones.

Every backbone maps a batch of uint8 RGB frames ``(N, H, W, 3)`` to pooled
embeddings ``(N, D)`` (float32, on CPU). Backbones are frozen; no grad.

Available:
  vl3-siglip   DAMO-NLP-SG/VL3-SigLIP-NaViT  -- VideoLLaMA3 vision tower (SigLIP so400m). Default.
  siglip2      google/siglip2-so400m-patch14-384 image tower
  dinov2       facebook/dinov2-large  (pure-vision SSL baseline, no language)
  i3d-compat   passthrough loader for existing pre-extracted .npy (sanity / parity runs)

Add InternVideo2 / VideoLLaMA3-full by implementing another `FrameBackbone`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import torch

_REGISTRY: dict[str, type["FrameBackbone"]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.backbone_name = name
        return cls

    return deco


def list_backbones() -> list[str]:
    return sorted(_REGISTRY)


def build_backbone(name: str, device: str = "cuda", dtype: str = "bf16", **kw) -> "FrameBackbone":
    if name not in _REGISTRY:
        raise ValueError(f"unknown backbone {name!r}; available: {list_backbones()}")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    return _REGISTRY[name](device=device, torch_dtype=torch_dtype, **kw)


@dataclass
class FrameBackbone(abc.ABC):
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    batch_size: int = 64
    backbone_name: str = "abstract"

    def __post_init__(self):
        self._build()

    @abc.abstractmethod
    def _build(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        ...

    @abc.abstractmethod
    def _forward(self, frames_uint8: np.ndarray) -> torch.Tensor:
        """frames_uint8: (N,H,W,3) -> (N, D) float tensor on self.device."""

    @torch.no_grad()
    def encode(self, frames_uint8: np.ndarray) -> np.ndarray:
        outs = []
        for s in range(0, len(frames_uint8), self.batch_size):
            chunk = frames_uint8[s : s + self.batch_size]
            outs.append(self._forward(chunk).float().cpu())
        return torch.cat(outs, 0).numpy().astype(np.float32)


# --------------------------------------------------------------------------------------
# HF image-tower backbones
# --------------------------------------------------------------------------------------


class _HFImageTower(FrameBackbone):
    model_id: str = ""
    pool: str = "mean"  # "mean" over tokens or "cls"

    def _build(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()
        self._dim = self._infer_dim()

    def _infer_dim(self) -> int:
        cfg = self.model.config
        for attr in ("hidden_size", "vision_hidden_size", "projection_dim"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        # probe with a dummy frame
        dummy = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        return int(self._forward(dummy).shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def _pool(self, out) -> torch.Tensor:
        # out: transformers ModelOutput. Prefer last_hidden_state tokens.
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            tok = out.last_hidden_state  # (N, L, D)
            if self.pool == "cls":
                return tok[:, 0]
            return tok.mean(dim=1)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if isinstance(out, torch.Tensor):
            return out.mean(dim=1) if out.dim() == 3 else out
        raise RuntimeError(f"cannot pool output of type {type(out)}")

    def _forward(self, frames_uint8: np.ndarray) -> torch.Tensor:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames_uint8]
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.torch_dtype):
            out = self.model(**inputs)
        return self._pool(out)


@register("vl3-siglip")
class VL3SigLIP(_HFImageTower):
    """VideoLLaMA3 vision tower. Mean-pooled patch tokens (~1152-d)."""

    model_id = "DAMO-NLP-SG/VL3-SigLIP-NaViT"
    pool = "mean"

    def _forward(self, frames_uint8: np.ndarray) -> torch.Tensor:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames_uint8]
        # VL3 processor expects merge_size; keep native patch grid (merge_size=1)
        inputs = self.processor(images=imgs, merge_size=1, return_tensors="pt")
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.torch_dtype):
            out = self.model(**inputs)
        feats = out[0] if isinstance(out, (tuple, list)) else self._pool(out)
        if feats.dim() == 3:               # (N, L, D) -> mean over tokens
            feats = feats.mean(dim=1)
        elif feats.dim() == 2 and feats.shape[0] != len(imgs):
            # NaViT returns flat (sum_L, D); split by patches-per-image if provided
            n = len(imgs)
            feats = feats.view(n, -1, feats.shape[-1]).mean(dim=1)
        return feats


@register("siglip2")
class SigLIP2(_HFImageTower):
    """SigLIP2 image tower via ``get_image_features`` -- the projected embedding
    in the *shared* contrastive space (pairs with the siglip2 text tower). The
    raw ``vision_model`` token mean is NOT in the text space -- do not use it."""

    model_id = "google/siglip2-so400m-patch14-384"

    def _build(self) -> None:
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype
        ).to(self.device)
        self.model.eval()
        self._dim = int(self.model.config.text_config.hidden_size)

    def _forward(self, frames_uint8: np.ndarray) -> torch.Tensor:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames_uint8]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.torch_dtype):
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        if not torch.is_tensor(feats):                    # transformers>=5: returns an output
            feats = feats.pooler_output
        return feats


@register("dinov2")
class DINOv2(_HFImageTower):
    model_id = "facebook/dinov2-large"
    pool = "cls"


# --------------------------------------------------------------------------------------
# passthrough (parity runs against existing I3D)
# --------------------------------------------------------------------------------------


@register("i3d-compat")
class I3DCompat(FrameBackbone):
    """Not a real extractor -- used only to validate the pipeline plumbing end to
    end against pre-extracted features. `encode` is unused; the CLI special-cases
    this name to copy/resample existing .npy files."""

    def _build(self) -> None:  # noqa: D401
        self._dim = 2048

    @property
    def dim(self) -> int:
        return self._dim

    def _forward(self, frames_uint8: np.ndarray) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("i3d-compat is passthrough only")
