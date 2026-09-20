"""Hybrid Temporal Alignment (TA): frame x action similarity -> monotonic coarse
alignment -> two-branch boundary refinement -> frame-level pseudo-labels Y*.

Decoupled from the WLTA training code: numpy/torch in, tensors out. Built on the
unit-tested primitives in `delta.align` (DP / ASOT decode, local boundary search).
"""
from .similarity import FrameActionSimilarity, cosine_similarity
from .coarse_align import coarse_align
from .boundary_refinement import BranchResult, branch_a_semantic, branch_b_relational
from .pseudo_labels import HybridConfig, HybridResult, fuse_boundaries, generate_pseudo_labels, generate_batch

__all__ = [
    "FrameActionSimilarity", "cosine_similarity", "coarse_align",
    "BranchResult", "branch_a_semantic", "branch_b_relational",
    "HybridConfig", "HybridResult", "fuse_boundaries", "generate_pseudo_labels", "generate_batch",
]
