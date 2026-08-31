"""Visualization helpers (matplotlib). Segmentation-timeline plotting for
comparing ground truth, alignment pseudo-labels, and predictions.
"""

from .timeline import class_colors, label_strip, plot_segmentation, segments_text

__all__ = ["class_colors", "label_strip", "plot_segmentation", "segments_text"]
