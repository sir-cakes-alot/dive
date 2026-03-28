"""
DIVE — Data Insights & Visualization Experience

A clean, ergonomic container for ordered numeric data with built-in
statistical analysis, ensemble prediction, and ASCII visualisation.

Requires Python ≥ 3.10
"""

from __future__ import annotations

from typing import Iterable, Iterator, Union, Callable, Any

from dive_core import DiveCore
from dive_stats import DiveStats
from dive_predict import DivePredict
from dive_viz import DiveViz
from dive_export import DiveExport
from dive_transforms import DiveTransforms

Numeric = Union[int, float]


class Dive(DiveCore, DiveStats, DivePredict, DiveViz, DiveExport, DiveTransforms):
    """Ordered numeric container with built-in statistics, prediction and viz."""
    __slots__ = ()

    # A simple wrapper class: detailed behavior is in the mixins.
    pass


if __name__ == "__main__":
    # Simple sanity tests
    d = Dive([3, 1, 4, 1, 5, 9, 2, 6])
    print("mean", d.mean())
    print("median", d.median())
    print("sum", d.sum())
    d += 7
    print("predict_next", d.predict_next())
    print(d.summary())
