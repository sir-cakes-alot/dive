"""
DIVE Core — Basic data container and management.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, Union, Any

from .stats import DiveStats
from .viz import DiveViz
from .transforms import DiveTransforms
from .predict import DivePredict
from .export import DiveExport

Numeric = Union[int, float]


class Dive(DiveStats, DiveViz, DiveTransforms, DivePredict, DiveExport):
    """Core Dive class with data management and basic operations."""

    __slots__ = ("_data",)

    def __init__(self, data: Iterable[Numeric] | Numeric | None = None) -> None:
        self._data: list[float] = []
        if data is not None:
            self.add(data)

    def add(self, *values: Numeric | Iterable[Numeric]) -> Dive:
        """Append one or more values (or iterables of values).

        Returns *self* so calls can be chained.
        Synonyms: :meth:`append`, ``+=``.
        """
        for v in values:
            if isinstance(v, (int, float)):
                self._data.append(float(v))
            elif isinstance(v, Iterable):
                self._data.extend(float(x) for x in v)
            else:
                self._data.append(float(v))
        return self

    append = add

    def remove(self, value: Numeric) -> Dive:
        """Remove the first occurrence of *value*."""
        self._data.remove(float(value))
        return self

    def pop(self, index: int = -1) -> float:
        """Remove and return the item at *index* (default last)."""
        return self._data.pop(index)

    def clear(self) -> Dive:
        """Remove all data points."""
        self._data.clear()
        return self

    def copy(self) -> Dive:
        """Return a shallow copy."""
        clone = Dive()
        clone._data = list(self._data)
        return clone

    @property
    def data(self) -> list[float]:
        """A copy of the internal data list."""
        return list(self._data)

    @data.setter
    def data(self, values: Iterable[Numeric]) -> None:
        self._data = [float(x) for x in values]

    @property
    def count(self) -> int:
        """Number of data points (same as ``len(d)``)."""
        return len(self._data)

    def __repr__(self) -> str:
        if len(self._data) <= 8:
            inner = ", ".join(f"{x:g}" for x in self._data)
            return f"Dive([{inner}])"
        head = ", ".join(f"{x:g}" for x in self._data[:4])
        tail = ", ".join(f"{x:g}" for x in self._data[-2:])
        return f"Dive([{head}, …, {tail}]  n={len(self._data)})"

    __str__ = __repr__

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int | slice) -> float | Dive:
        result = self._data[index]
        return Dive(result) if isinstance(index, slice) else result

    def __setitem__(self, index: int, value: Numeric) -> None:
        self._data[index] = float(value)

    def __delitem__(self, index: int | slice) -> None:
        del self._data[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __contains__(self, item: Numeric) -> bool:
        return float(item) in self._data

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dive):
            return self._data == other._data
        return NotImplemented

    def __iadd__(self, other: Numeric | Iterable[Numeric]) -> Dive:
        return self.add(other)

    def __add__(self, other: Dive | Iterable[Numeric]) -> Dive:
        clone = self.copy()
        clone.add(other._data if isinstance(other, Dive) else other)
        return clone

    def _require(self, n: int = 1) -> None:
        """Raise if the dataset has fewer than *n* points."""
        if len(self._data) < n:
            raise ValueError(
                f"Operation requires >= {n} data point(s); "
                f"dataset contains {len(self._data)}."
            )

    @staticmethod
    def _is_nearly_zero(val: float, tol: float = 1e-9) -> bool:
        return abs(val) < tol

    @staticmethod
    def _is_nearly_constant(seq: list[float], tol: float = 1e-9) -> bool:
        if not seq:
            return True
        ref = seq[0]
        return all(abs(v - ref) < tol for v in seq)

    @staticmethod
    def _is_nearly_equal(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) < tol

    @staticmethod
    def _round_if_close(val: float, tol: float = 1e-9) -> float:
        rounded = round(val)
        if abs(val - rounded) < tol:
            return float(rounded)
        for decimals in range(1, 10):
            r = round(val, decimals)
            if abs(val - r) < tol:
                return r
        return val

    @staticmethod
    def _safe_div(a: float, b: float, default: float = float("inf")) -> float:
        """Safe division that returns default on zero/near-zero divisor."""
        if abs(b) < 1e-15:
            return default
        return a / b

    @staticmethod
    def _safe_call(func: callable, *args, default: Any = None) -> Any:
        """Call function with args, returning default on any exception."""
        try:
            result = func(*args)
            if result is None or (isinstance(result, float) and (math.isnan(result) or math.isinf(result))):
                return default
            return result
        except Exception:
            return default