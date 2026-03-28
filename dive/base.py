from __future__ import annotations
import math
from typing import Iterable, Union, Any

Numeric = Union[int, float]

class DiveBase:
    """Base class for Dive with common data and utility methods."""
    
    __slots__ = ("_data",)

    def __init__(self, data: Iterable[Numeric] | Numeric | None = None) -> None:
        self._data: list[float] = []
        # In base class, we don't have .add, but core.Dive will override this.
        # However, to make mixins work, we should have a basic way to set data.
        if data is not None:
            if isinstance(data, (int, float)):
                self._data = [float(data)]
            elif isinstance(data, Iterable):
                self._data = [float(x) for x in data]
            else:
                self._data = [float(data)]

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
