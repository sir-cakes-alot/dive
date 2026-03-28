"""
DIVE Transforms — Data transformation and cross-dataset methods.
"""

import math
import statistics as _stats
from typing import Callable


from .stats import DiveStats


class DiveTransforms(DiveStats):
    """Data transformation methods for Dive."""

    def z_scores(self) -> list[float]:
        self._require(2)
        m, s = self.mean(), self.stdev()
        if s == 0:
            return [0.0] * len(self._data)
        return [(x - m) / s for x in self._data]

    def normalized(self) -> list[float]:
        self._require()
        lo, hi = self.min(), self.max()
        r = hi - lo
        if r == 0:
            return [0.5] * len(self._data)
        return [(x - lo) / r for x in self._data]

    def cumulative_sum(self) -> list[float]:
        out: list[float] = []
        total = 0.0
        for x in self._data:
            total += x
            out.append(total)
        return out

    def moving_average(self, window: int = 3) -> list[float]:
        if window < 1:
            raise ValueError("Window must be >= 1.")
        self._require(window)
        return [
            math.fsum(self._data[i : i + window]) / window
            for i in range(len(self._data) - window + 1)
        ]

    def diff(self, periods: int = 1) -> 'DiveTransforms':
        """Return first differences (or n-th differences if periods > 1)."""
        if periods < 1:
            raise ValueError("periods must be >= 1.")
        result = list(self._data)
        for _ in range(periods):
            result = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        from .core import Dive
        return Dive(result)

    def pct_change(self) -> list[float]:
        """Percentage change from previous value."""
        self._require(2)
        return [
            self._safe_div(self._data[i] - self._data[i - 1], self._data[i - 1], 0.0)
            for i in range(1, len(self._data))
        ]

    def sorted(self, *, reverse: bool = False) -> list[float]:
        return sorted(self._data, reverse=reverse)

    def clip(self, lo: float | None = None, hi: float | None = None) -> 'DiveTransforms':
        """Clip values to [lo, hi] range."""
        result = []
        for x in self._data:
            if lo is not None and x < lo:
                x = lo
            if hi is not None and x > hi:
                x = hi
            result.append(x)
        from .core import Dive
        return Dive(result)

    def apply(self, func: Callable[[float], float]) -> 'DiveTransforms':
        """Apply a function to each element."""
        from .core import Dive
        return Dive([func(x) for x in self._data])

    def outliers(self, method: str = "iqr", k: float = 1.5) -> list[float]:
        if method == "iqr":
            self._require(4)
            q1, _, q3 = self.quartiles()
            iqr_val = q3 - q1
            lo, hi = q1 - k * iqr_val, q3 + k * iqr_val
            return [x for x in self._data if x < lo or x > hi]
        if method == "zscore":
            self._require(2)
            zs = self.z_scores()
            return [x for x, z in zip(self._data, zs) if abs(z) > k]
        raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

    def correlation(self, other: 'DiveTransforms', *, outliers: float = 0) -> float:
        if not isinstance(other, DiveTransforms):
            raise TypeError("correlation() requires another Dive instance.")
        if len(self._data) != len(other._data):
            raise ValueError("Both datasets must have the same length.")
        self._require(2)

        if not 0 <= outliers < 1:
            raise ValueError("outliers must be >= 0 and < 1.")

        xs: list[float] = self._data
        ys: list[float] = other._data

        if outliers > 0:
            p_lo = outliers * 100
            p_hi = 100 - p_lo
            x_lo, x_hi = self.percentile(p_lo), self.percentile(p_hi)
            y_lo, y_hi = other.percentile(p_lo), other.percentile(p_hi)
            pairs = [
                (x, y)
                for x, y in zip(xs, ys)
                if x_lo <= x <= x_hi and y_lo <= y <= y_hi
            ]
            if len(pairs) < 2:
                raise ValueError(
                    f"Only {len(pairs)} pair(s) survived trimming — need at least 2."
                )
            xs, ys = zip(*pairs)

        n = len(xs)
        mx = math.fsum(xs) / n
        my = math.fsum(ys) / n
        sx = math.sqrt(math.fsum((x - mx) ** 2 for x in xs) / (n - 1))
        sy = math.sqrt(math.fsum((y - my) ** 2 for y in ys) / (n - 1))

        if sx == 0 or sy == 0:
            raise ValueError("Cannot compute correlation when std dev is zero.")

        cov = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
        return cov / (sx * sy)

    def covariance(self, other: 'DiveTransforms') -> float:
        if not isinstance(other, DiveTransforms):
            raise TypeError("covariance() requires another Dive instance.")
        n = len(self._data)
        if n != len(other._data):
            raise ValueError("Both datasets must have the same length.")
        self._require(2)
        mx, my = self.mean(), _stats.mean(other._data)
        return math.fsum(
            (x - mx) * (y - my) for x, y in zip(self._data, other._data)
        ) / (n - 1)

    def regress_on(self, other: 'DiveTransforms') -> tuple[float, float, float]:
        """
        Simple linear regression: self = a + b * other.

        Returns (intercept, slope, r_squared).
        """
        if len(self._data) != len(other._data):
            raise ValueError("Both datasets must have the same length.")
        self._require(2)

        xs = other._data
        ys = self._data
        n = len(xs)

        x_mean = math.fsum(xs) / n
        y_mean = math.fsum(ys) / n

        num = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = math.fsum((x - x_mean) ** 2 for x in xs)

        if abs(den) < 1e-15:
            return y_mean, 0.0, 0.0

        slope = num / den
        intercept = y_mean - slope * x_mean

        # R-squared
        ss_tot = math.fsum((y - y_mean) ** 2 for y in ys)
        ss_res = math.fsum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))

        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return intercept, slope, r_sq