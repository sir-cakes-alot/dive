"""
DIVE — Data Insights & Visualization Experience

A clean, ergonomic container for ordered numeric data with built-in
statistical analysis, ensemble prediction, and ASCII visualisation.

Requires Python ≥ 3.10
"""

from __future__ import annotations

import math
import time
import statistics as _stats
from typing import Iterable, Iterator, Union, Callable, Any

from dive_core import DiveCore
from dive_stats import DiveStats
from dive_predict import DivePredict
from dive_viz import DiveViz
from dive_export import DiveExport
from dive_transforms import DiveTransforms

Numeric = Union[int, float]


class Dive(DiveCore, DiveStats, DivePredict, DiveViz, DiveExport, DiveTransforms):
    """Ordered numeric container with built-in statistics and visualisation.

    Examples
    --------
    >>> d = Dive([3, 1, 4, 1, 5, 9, 2, 6])
    >>> d.mean()
    3.875
    >>> d += 7
    >>> d.median()
    4.0
    >>> d.predict_next()          # inverse-error weighted ensemble
    8.066...
    >>> print(d.summary())
    
    Regression prediction:
    >>> sales = Dive([100, 150, 120, 200, 180])
    >>> temps = Dive([20, 25, 22, 30, 28, 35])  # includes tomorrow's temp
    >>> sales.predict_next(reference=temps, TA=1)  # predicts sales for temp=35
    """

    __slots__ = ()

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

    # ================================================================== #
    #  Properties                                                         #
    # ================================================================== #

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

    # ================================================================== #
    #  Protocol / Dunder Methods                                          #
    # ================================================================== #

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

    # ================================================================== #
    #  Internal Helpers                                                   #
    # ================================================================== #

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
    def _safe_call(func: Callable, *args, default: Any = None) -> Any:
        """Call function with args, returning default on any exception."""
        try:
            result = func(*args)
            if result is None or (isinstance(result, float) and (math.isnan(result) or math.isinf(result))):
                return default
            return result
        except Exception:
            return default

    # ================================================================== #
    #  Central Tendency                                                   #
    # ================================================================== #

    def mean(self) -> float:
        self._require()
        return _stats.mean(self._data)

    def median(self) -> float:
        self._require()
        return _stats.median(self._data)

    def mode(self) -> list[float]:
        self._require()
        return _stats.multimode(self._data)

    def geo_mean(self, filter_bad: bool = False) -> float:
        self._require()
        if not filter_bad:
            if any(x <= 0 for x in self._data):
                raise ValueError("Geometric mean requires all strictly positive values.")
        else:
            temp_dive = Dive([i if i > 0 else 0.0001 for i in self._data])
            return temp_dive.geo_mean()
        return math.exp(
            math.fsum(math.log(x) for x in self._data) / len(self._data)
        )

    def harmonic_mean(self, filter_bad: bool = False) -> float:
        self._require()
        if not filter_bad:
            if any(x <= 0 for x in self._data):
                raise ValueError("Harmonic mean requires all strictly positive values.")
        else:
            temp_dive = Dive([i if i > 0 else 0.0001 for i in self._data])
            return temp_dive.harmonic_mean()
        return _stats.harmonic_mean(self._data)

    # ================================================================== #
    #  Dispersion                                                         #
    # ================================================================== #

    def stdev(self, population: bool = False) -> float:
        self._require(1 if population else 2)
        return (_stats.pstdev if population else _stats.stdev)(self._data)

    def variance(self, population: bool = False) -> float:
        self._require(1 if population else 2)
        return (_stats.pvariance if population else _stats.variance)(self._data)

    def range(self) -> float:
        self._require()
        return max(self._data) - min(self._data)

    def iqr(self) -> float:
        return self.percentile(75) - self.percentile(25)

    def min(self) -> float:
        self._require()
        return min(self._data)

    def max(self) -> float:
        self._require()
        return max(self._data)

    def sum(self) -> float:
        return math.fsum(self._data)

    # ================================================================== #
    #  Percentiles & Quantiles                                            #
    # ================================================================== #

    def percentile(self, p: float) -> float:
        self._require()
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100.")
        s = sorted(self._data)
        k = (p / 100) * (len(s) - 1)
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        if lo == hi:
            return s[lo]
        return s[lo] + (k - lo) * (s[hi] - s[lo])

    def quartiles(self) -> tuple[float, float, float]:
        return self.percentile(25), self.percentile(50), self.percentile(75)

    # ================================================================== #
    #  Distribution Shape                                                 #
    # ================================================================== #

    def skewness(self) -> float:
        self._require(3)
        n = len(self._data)
        m, s = self.mean(), self.stdev()
        if s == 0:
            return 0.0
        adj = n / ((n - 1) * (n - 2))
        return adj * math.fsum(((x - m) / s) ** 3 for x in self._data)

    def kurtosis(self) -> float:
        self._require(4)
        n = len(self._data)
        m = self.mean()
        s = self.stdev(population=True)
        if s == 0:
            return 0.0
        return math.fsum(((x - m) / s) ** 4 for x in self._data) / n - 3.0

    # ================================================================== #
    #  Time-Series Pattern Detection (Private)                            #
    # ================================================================== #

    def _detect_polynomial_order(self) -> int | None:
        n = len(self._data)
        if n < 2:
            return None
        max_order = min(n - 1, 10)
        current = list(self._data)
        for order in range(max_order):
            if self._is_nearly_constant(current):
                return order
            next_diffs = [current[i + 1] - current[i] for i in range(len(current) - 1)]
            if len(next_diffs) < 1:
                return None
            current = next_diffs
        if self._is_nearly_constant(current):
            return max_order
        return None

    def _predict_polynomial_exact(self, steps: int = 1) -> list[float] | None:
        n = len(self._data)
        if n < 2:
            return None
        order = self._detect_polynomial_order()
        if order is None:
            return None

        diff_table: list[list[float]] = [list(self._data)]
        for k in range(order):
            prev = diff_table[-1]
            diff_table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])

        if diff_table[-1]:
            raw_const = math.fsum(diff_table[-1]) / len(diff_table[-1])
            const_val = self._round_if_close(raw_const)
        else:
            const_val = 0.0

        deltas = [row[0] for row in diff_table]
        if len(deltas) > order:
            deltas[order] = const_val
        for i in range(len(deltas)):
            deltas[i] = self._round_if_close(deltas[i])

        def _eval_newton(t: float) -> float:
            total = 0.0
            falling = 1.0
            for k in range(len(deltas)):
                if k == 0:
                    falling = 1.0
                else:
                    falling *= (t - (k - 1)) / k
                total += falling * deltas[k]
            return total

        results = []
        for step in range(1, steps + 1):
            t = n - 1 + step
            raw = _eval_newton(float(t))
            results.append(self._round_if_close(raw))
        return results

    def _detect_geometric(self) -> float | None:
        n = len(self._data)
        if n < 3:
            return None
        if any(x == 0 for x in self._data):
            return None
        ratios = [self._data[i + 1] / self._data[i] for i in range(n - 1)]
        if self._is_nearly_constant(ratios, tol=1e-9):
            return math.fsum(ratios) / len(ratios)
        return None

    def _predict_geometric(self, steps: int = 1) -> list[float] | None:
        ratio = self._detect_geometric()
        if ratio is None:
            return None
        last = self._data[-1]
        results = []
        for i in range(1, steps + 1):
            val = last * (ratio ** i)
            results.append(self._round_if_close(val))
        return results

    def _detect_ratio_pattern(self) -> tuple[str, tuple] | None:
        n = len(self._data)
        if n < 4:
            return None
        diffs = [self._data[i + 1] - self._data[i] for i in range(n - 1)]
        if len(diffs) >= 3 and all(d != 0 for d in diffs):
            ratios = [diffs[i + 1] / diffs[i] for i in range(len(diffs) - 1)]
            if self._is_nearly_constant(ratios, tol=1e-9):
                avg_ratio = math.fsum(ratios) / len(ratios)
                return ("diff_geometric", (diffs[-1], avg_ratio))
        if len(diffs) >= 4:
            diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            if len(diffs2) >= 3 and all(d != 0 for d in diffs2):
                ratios2 = [diffs2[i + 1] / diffs2[i] for i in range(len(diffs2) - 1)]
                if self._is_nearly_constant(ratios2, tol=1e-9):
                    avg_ratio2 = math.fsum(ratios2) / len(ratios2)
                    return ("diff2_geometric", (diffs[-1], diffs2[-1], avg_ratio2))
        return None

    def _predict_ratio_pattern(self, steps: int = 1) -> list[float] | None:
        result = self._detect_ratio_pattern()
        if result is None:
            return None
        kind, params = result
        predictions = []
        last_val = self._data[-1]

        if kind == "diff_geometric":
            last_diff, ratio = params
            for _ in range(steps):
                last_diff *= ratio
                last_val += last_diff
                predictions.append(self._round_if_close(last_val))
        elif kind == "diff2_geometric":
            n = len(self._data)
            diffs = [self._data[i + 1] - self._data[i] for i in range(n - 1)]
            diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            last_diff = diffs[-1]
            last_diff2 = diffs2[-1]
            _, _, ratio = params
            for _ in range(steps):
                last_diff2 *= ratio
                last_diff += last_diff2
                last_val += last_diff
                predictions.append(self._round_if_close(last_val))
        return predictions if predictions else None

    # ================================================================== #
    #  Time-Series Prediction Models (Private)                            #
    # ================================================================== #

    def _predict_lagrange(self, index: float) -> float:
        n = len(self._data)
        result = 0.0
        for i in range(n):
            term = self._data[i]
            for j in range(n):
                if i != j:
                    term *= (index - j) / (i - j)
            result += term
        return self._round_if_close(result)

    def _predict_newton(self, steps: int = 1) -> list[float]:
        n = len(self._data)
        table: list[list[float]] = [list(self._data)]
        for k in range(1, n):
            prev = table[-1]
            row = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
            if not row:
                break
            table.append(row)
            if all(self._is_nearly_zero(v) for v in row):
                break

        tails = [row[-1] for row in table]
        for i in range(len(tails)):
            tails[i] = self._round_if_close(tails[i])

        effective_order = len(tails) - 1
        while effective_order > 0 and self._is_nearly_zero(tails[effective_order]):
            effective_order -= 1
        keep = max(effective_order + 1, min(2, len(tails)))
        tails = tails[:keep]

        results = []
        for _ in range(steps):
            for i in range(len(tails) - 2, -1, -1):
                tails[i] = tails[i] + tails[i + 1]
            results.append(self._round_if_close(tails[0]))
        return results

    def linear_regression(self) -> tuple[float, float]:
        self._require(2)
        n = len(self._data)
        x_mean = (n - 1) / 2.0
        y_mean = self.mean()
        num = math.fsum((i - x_mean) * (y - y_mean) for i, y in enumerate(self._data))
        den = math.fsum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return 0.0, y_mean
        slope = num / den
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def _predict_linear(self, index: float) -> float:
        slope, intercept = self.linear_regression()
        return slope * index + intercept

    def _fit_quadratic(self) -> tuple[float, float, float]:
        n = len(self._data)
        xs = range(n)
        s_x = math.fsum(xs)
        s_x2 = math.fsum(x ** 2 for x in xs)
        s_x3 = math.fsum(x ** 3 for x in xs)
        s_x4 = math.fsum(x ** 4 for x in xs)
        s_y = math.fsum(self._data)
        s_xy = math.fsum(x * y for x, y in zip(xs, self._data))
        s_x2y = math.fsum(x ** 2 * y for x, y in zip(xs, self._data))

        mat = [
            [float(n), s_x, s_x2, s_y],
            [s_x, s_x2, s_x3, s_xy],
            [s_x2, s_x3, s_x4, s_x2y],
        ]
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(mat[r][col]))
            mat[col], mat[pivot] = mat[pivot], mat[col]
            if abs(mat[col][col]) < 1e-15:
                s, i = self.linear_regression()
                return i, s, 0.0
            for row in range(col + 1, 3):
                f = mat[row][col] / mat[col][col]
                for j in range(col, 4):
                    mat[row][j] -= f * mat[col][j]

        c = [0.0] * 3
        for i in range(2, -1, -1):
            c[i] = mat[i][3]
            for j in range(i + 1, 3):
                c[i] -= mat[i][j] * c[j]
            c[i] /= mat[i][i]
        return c[0], c[1], c[2]

    def _predict_quadratic(self, index: float) -> float:
        a, b, c = self._fit_quadratic()
        return a + b * index + c * index ** 2

    def _fit_polynomial(self, degree: int) -> list[float]:
        n = len(self._data)
        if degree >= n:
            degree = n - 1
        d = degree + 1
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d

        powers = []
        for i in range(n):
            row = [1.0]
            for _ in range(1, 2 * degree + 1):
                row.append(row[-1] * i)
            powers.append(row)

        for j in range(d):
            for k in range(d):
                xtx[j][k] = math.fsum(powers[i][j + k] for i in range(n))
            xty[j] = math.fsum(powers[i][j] * self._data[i] for i in range(n))

        aug = [xtx[r][:] + [xty[r]] for r in range(d)]
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-15:
                if degree > 1:
                    return self._fit_polynomial(degree - 1) + [0.0]
                s, intercept = self.linear_regression()
                return [intercept, s] + [0.0] * (degree - 1)
            for row in range(col + 1, d):
                f = aug[row][col] / aug[col][col]
                for j in range(col, d + 1):
                    aug[row][j] -= f * aug[col][j]

        coeffs = [0.0] * d
        for i in range(d - 1, -1, -1):
            coeffs[i] = aug[i][d]
            for j in range(i + 1, d):
                coeffs[i] -= aug[i][j] * coeffs[j]
            coeffs[i] /= aug[i][i]
        return coeffs

    def _predict_polynomial(self, degree: int, index: float) -> float:
        coeffs = self._fit_polynomial(degree)
        result = 0.0
        xi = 1.0
        for c in coeffs:
            result += c * xi
            xi *= index
        return self._round_if_close(result)

    def _predict_holt(self, steps: int) -> list[float]:
        if len(self._data) < 2:
            return [self._data[-1]] * steps if self._data else [0.0] * steps

        best, best_mse = (0.3, 0.1), float("inf")
        for ai in range(1, 100, 5):
            a = ai / 100
            for bi in range(1, 60, 5):
                b = bi / 100
                level = self._data[0]
                trend = self._data[1] - self._data[0]
                sse = 0.0
                for t in range(1, len(self._data)):
                    sse += (self._data[t] - (level + trend)) ** 2
                    nl = a * self._data[t] + (1 - a) * (level + trend)
                    trend = b * (nl - level) + (1 - b) * trend
                    level = nl
                mse = sse / (len(self._data) - 1)
                if mse < best_mse:
                    best_mse, best = mse, (a, b)

        alpha, beta = best
        level = self._data[0]
        trend = self._data[1] - self._data[0]
        for t in range(1, len(self._data)):
            new_level = alpha * self._data[t] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level, trend = new_level, new_trend

        return [level + h * trend for h in range(1, steps + 1)]

    def _predict_exponential(self, index: float) -> float | None:
        if any(y <= 0 for y in self._data):
            return None
        n = len(self._data)
        x_mean = (n - 1) / 2.0
        log_y = [math.log(y) for y in self._data]
        y_mean = math.fsum(log_y) / n
        num = math.fsum((i - x_mean) * (ly - y_mean) for i, ly in enumerate(log_y))
        den = math.fsum((i - x_mean) ** 2 for i in range(n))
        if den == 0:
            return math.exp(y_mean)
        slope = num / den
        intercept = y_mean - slope * x_mean
        return math.exp(slope * index + intercept)

    def _predict_differences(self, steps: int) -> list[float]:
        n = len(self._data)
        order = min(n - 1, 4)
        tails = [self._data[-1]]
        current_seq = self._data[-(order + 1):]
        for _ in range(order):
            next_seq = [current_seq[i + 1] - current_seq[i] for i in range(len(current_seq) - 1)]
            tails.append(next_seq[-1])
            current_seq = next_seq

        preds = []
        for _ in range(steps):
            for i in range(order - 1, -1, -1):
                tails[i] += tails[i + 1]
            preds.append(tails[0])
        return preds

    def _predict_drift(self, steps: int) -> list[float]:
        n = len(self._data)
        slope = (self._data[-1] - self._data[0]) / (n - 1) if n > 1 else 0.0
        return [self._data[-1] + slope * i for i in range(1, steps + 1)]

    def _predict_seasonal_naive(self, steps: int, period: int | None = None) -> list[float]:
        """Seasonal naive: repeat the pattern from `period` steps ago."""
        n = len(self._data)
        if period is None:
            period = self._detect_seasonality()
        if period is None or period < 2 or period >= n:
            return self._predict_drift(steps)

        results = []
        for s in range(steps):
            idx = n - period + (s % period)
            results.append(self._data[idx])
        return results

    def _detect_seasonality(self) -> int | None:
        """Detect seasonality via autocorrelation peaks."""
        n = len(self._data)
        if n < 6:
            return None

        mean_val = self.mean()
        var_val = self.variance(population=True)
        if var_val == 0:
            return None

        best_period = None
        best_acf = 0.3  # minimum threshold

        for lag in range(2, n // 2):
            acf = math.fsum(
                (self._data[i] - mean_val) * (self._data[i - lag] - mean_val)
                for i in range(lag, n)
            ) / ((n - lag) * var_val)

            if acf > best_acf:
                best_acf = acf
                best_period = lag

        return best_period

    # ================================================================== #
    #  Ensemble Prediction Engine (Private)                               #
    # ================================================================== #

    def _backtest(self, k: int) -> dict[str, float]:
        n = len(self._data)
        errors: dict[str, list[float]] = {
            "linear": [], "quadratic": [], "holt": [],
            "exponential": [], "differences": [], "drift": [],
            "newton": [], "lagrange": [], "poly_auto": [],
            "poly_3": [], "poly_4": [], "poly_5": [],
            "seasonal": [],
        }

        for offset in range(k):
            split = n - k + offset
            if split < 2:
                continue
            sub = Dive(self._data[:split])
            actual = self._data[split]

            for name, func in [
                ("linear", lambda s=sub, sp=split: s._predict_linear(sp)),
                ("newton", lambda s=sub: s._predict_newton(1)[0]),
                ("lagrange", lambda s=sub, sp=split: s._predict_lagrange(sp)),
                ("differences", lambda s=sub: s._predict_differences(1)[0]),
                ("drift", lambda s=sub: s._predict_drift(1)[0]),
            ]:
                pred = self._safe_call(func)
                if pred is not None:
                    errors[name].append(abs(actual - pred))

            pred = self._safe_call(lambda s=sub, sp=split: s._predict_exponential(sp))
            if pred is not None:
                errors["exponential"].append(abs(actual - pred))

            pred = self._safe_call(lambda s=sub: s._predict_seasonal_naive(1)[0])
            if pred is not None:
                errors["seasonal"].append(abs(actual - pred))

            poly_order = sub._detect_polynomial_order()
            deg = poly_order if poly_order is not None else min(3, split - 1)
            pred = self._safe_call(lambda s=sub, d=deg, sp=split: s._predict_polynomial(d, sp))
            if pred is not None:
                errors["poly_auto"].append(abs(actual - pred))

            for deg in [3, 4, 5]:
                if split > deg:
                    pred = self._safe_call(lambda s=sub, d=deg, sp=split: s._predict_polynomial(d, sp))
                    if pred is not None:
                        errors[f"poly_{deg}"].append(abs(actual - pred))

            if split >= 3:
                pred = self._safe_call(lambda s=sub, sp=split: s._predict_quadratic(sp))
                if pred is not None:
                    errors["quadratic"].append(abs(actual - pred))
                pred = self._safe_call(lambda s=sub: s._predict_holt(1)[0])
                if pred is not None:
                    errors["holt"].append(abs(actual - pred))

        return {m: (math.fsum(e) / len(e) if e else float("inf")) for m, e in errors.items()}

    @staticmethod
    def _inverse_weights(maes: dict[str, float]) -> dict[str, float]:
        eps = 1e-15
        finite = {m: v for m, v in maes.items() if v < float("inf")}
        if not finite:
            return {m: 1.0 / len(maes) for m in maes}

        min_mae = min(finite.values())
        if min_mae < 1e-12:
            perfect = {m for m, v in finite.items() if v < 1e-12}
            if perfect:
                w = 1.0 / len(perfect)
                return {m: (w if m in perfect else 0.0) for m in maes}

        inv = {m: 1.0 / (v + eps) for m, v in finite.items()}
        total = sum(inv.values())
        return {m: inv.get(m, 0.0) / total for m in maes}

    def _get_all_predictions(self, steps: int) -> dict[str, list[float]]:
        n = len(self._data)
        preds: dict[str, list[float]] = {}

        for name, func in [
            ("linear", lambda s=steps: [self._predict_linear(n - 1 + i) for i in range(1, s + 1)]),
            ("newton", lambda s=steps: self._predict_newton(s)),
            ("lagrange", lambda s=steps: [self._predict_lagrange(n - 1 + i) for i in range(1, s + 1)]),
            ("differences", lambda s=steps: self._predict_differences(s)),
            ("drift", lambda s=steps: self._predict_drift(s)),
            ("seasonal", lambda s=steps: self._predict_seasonal_naive(s)),
        ]:
            result = self._safe_call(func)
            if result is not None:
                preds[name] = result

        exp_preds = []
        for s in range(1, steps + 1):
            ep = self._safe_call(lambda idx=n - 1 + s: self._predict_exponential(idx))
            if ep is not None:
                exp_preds.append(ep)
        if len(exp_preds) == steps:
            preds["exponential"] = exp_preds

        poly_order = self._detect_polynomial_order()
        deg = poly_order if poly_order is not None else min(3, n - 1)
        result = self._safe_call(
            lambda d=deg, s=steps: [self._predict_polynomial(d, n - 1 + i) for i in range(1, s + 1)]
        )
        if result is not None:
            preds["poly_auto"] = result

        for deg in [3, 4, 5]:
            if n > deg:
                result = self._safe_call(
                    lambda d=deg, s=steps: [self._predict_polynomial(d, n - 1 + i) for i in range(1, s + 1)]
                )
                if result is not None:
                    preds[f"poly_{deg}"] = result

        if n >= 3:
            result = self._safe_call(
                lambda s=steps: [self._predict_quadratic(n - 1 + i) for i in range(1, s + 1)]
            )
            if result is not None:
                preds["quadratic"] = result
            result = self._safe_call(lambda s=steps: self._predict_holt(s))
            if result is not None:
                preds["holt"] = result

        return preds

    def _ensemble_predict(self, steps: int) -> list[float]:
        n = len(self._data)
        k = min(max(3, n // 3), 10)
        maes = self._backtest(k)
        weights = self._inverse_weights(maes)
        preds = self._get_all_predictions(steps)

        results = []
        for step_idx in range(steps):
            weighted_sum = 0.0
            weight_total = 0.0
            for model, pred_list in preds.items():
                if step_idx < len(pred_list) and model in weights:
                    w = weights[model]
                    weighted_sum += w * pred_list[step_idx]
                    weight_total += w
            if weight_total > 0:
                results.append(weighted_sum / weight_total)
            else:
                results.append(self._data[-1])
        return results

    # ================================================================== #
    #  Function Mapping Discovery (for Regression Mode)                   #
    # ================================================================== #

    @staticmethod
    def _fit_linear_xy(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
        """Fit y = a*x + b. Returns (a, b) or None."""
        n = len(xs)
        if n < 2 or n != len(ys):
            return None
        x_mean = math.fsum(xs) / n
        y_mean = math.fsum(ys) / n
        num = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = math.fsum((x - x_mean) ** 2 for x in xs)
        if abs(den) < 1e-15:
            return None
        a = num / den
        b = y_mean - a * x_mean
        return (a, b)

    @staticmethod
    def _fit_polynomial_xy(xs: list[float], ys: list[float], degree: int) -> list[float] | None:
        """Fit y = c0 + c1*x + c2*x^2 + ... Returns coefficients or None."""
        n = len(xs)
        if n < degree + 1 or n != len(ys):
            return None

        d = degree + 1
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d

        powers = []
        for x in xs:
            row = [1.0]
            for _ in range(1, 2 * degree + 1):
                row.append(row[-1] * x)
            powers.append(row)

        for j in range(d):
            for k in range(d):
                xtx[j][k] = math.fsum(powers[i][j + k] for i in range(n))
            xty[j] = math.fsum(powers[i][j] * ys[i] for i in range(n))

        aug = [xtx[r][:] + [xty[r]] for r in range(d)]
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-15:
                return None
            for row in range(col + 1, d):
                f = aug[row][col] / aug[col][col]
                for j in range(col, d + 1):
                    aug[row][j] -= f * aug[col][j]

        coeffs = [0.0] * d
        for i in range(d - 1, -1, -1):
            coeffs[i] = aug[i][d]
            for j in range(i + 1, d):
                coeffs[i] -= aug[i][j] * coeffs[j]
            if abs(aug[i][i]) < 1e-15:
                return None
            coeffs[i] /= aug[i][i]
        return coeffs

    def _discover_mappings(
        self,
        xs: list[float],
        ys: list[float],
        deadline: float,
        precision: int = 4
    ) -> list[tuple[Callable[[float], float], float, str]]:
        """
        Discover functions F such that F(xs[i]) ≈ ys[i].
        Returns list of (function, mean_abs_error, description) sorted by error.
        """
        if len(xs) != len(ys) or len(xs) < 2:
            return []

        candidates: list[tuple[Callable[[float], float], float, str]] = []

        def score(f: Callable[[float], float]) -> float:
            try:
                total = 0.0
                for x, y in zip(xs, ys):
                    pred = f(x)
                    if pred is None or math.isnan(pred) or math.isinf(pred):
                        return float("inf")
                    total += abs(pred - y)
                return total / len(xs)
            except Exception:
                return float("inf")

        def add_candidate(f: Callable, desc: str) -> None:
            err = score(f)
            if err < float("inf"):
                candidates.append((f, err, desc))

        # === Linear: y = a*x + b ===
        if time.time() < deadline:
            result = self._fit_linear_xy(xs, ys)
            if result:
                a, b = result
                a_r, b_r = round(a, precision), round(b, precision)
                add_candidate(lambda x, a=a_r, b=b_r: a * x + b, f"y = {a_r}*x + {b_r}")

        # === Polynomial degrees 2-6 ===
        for deg in range(2, 7):
            if time.time() >= deadline:
                break
            coeffs = self._fit_polynomial_xy(xs, ys, deg)
            if coeffs:
                coeffs_r = [round(c, precision) for c in coeffs]

                def make_poly(c: list[float]) -> Callable[[float], float]:
                    def f(x: float) -> float:
                        result = 0.0
                        xi = 1.0
                        for coef in c:
                            result += coef * xi
                            xi *= x
                        return result
                    return f

                f = make_poly(coeffs_r)
                terms = [f"{c}*x^{i}" if i > 0 else str(c) for i, c in enumerate(coeffs_r) if abs(c) > 1e-10]
                add_candidate(f, f"poly{deg}: {' + '.join(terms)}")

        # === Power: y = a * x^b ===
        if time.time() < deadline:
            pos_pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
            if len(pos_pairs) >= 2:
                log_xs = [math.log(x) for x, _ in pos_pairs]
                log_ys = [math.log(y) for _, y in pos_pairs]
                result = self._fit_linear_xy(log_xs, log_ys)
                if result:
                    b, log_a = result
                    a = math.exp(log_a)
                    a_r, b_r = round(a, precision), round(b, precision)
                    add_candidate(
                        lambda x, a=a_r, b=b_r: a * (x ** b) if x > 0 else float("nan"),
                        f"y = {a_r} * x^{b_r}"
                    )

        # === Exponential: y = a * e^(b*x) ===
        if time.time() < deadline:
            pos_ys = [(x, y) for x, y in zip(xs, ys) if y > 0]
            if len(pos_ys) >= 2:
                xs_pos = [x for x, _ in pos_ys]
                log_ys = [math.log(y) for _, y in pos_ys]
                result = self._fit_linear_xy(xs_pos, log_ys)
                if result:
                    b, log_a = result
                    a = math.exp(log_a)
                    a_r, b_r = round(a, precision), round(b, precision)
                    add_candidate(
                        lambda x, a=a_r, b=b_r: a * math.exp(b * x),
                        f"y = {a_r} * e^({b_r}*x)"
                    )

        # === Logarithmic: y = a * ln(x) + b ===
        if time.time() < deadline:
            pos_xs = [(x, y) for x, y in zip(xs, ys) if x > 0]
            if len(pos_xs) >= 2:
                log_xs = [math.log(x) for x, _ in pos_xs]
                ys_pos = [y for _, y in pos_xs]
                result = self._fit_linear_xy(log_xs, ys_pos)
                if result:
                    a, b = result
                    a_r, b_r = round(a, precision), round(b, precision)
                    add_candidate(
                        lambda x, a=a_r, b=b_r: a * math.log(x) + b if x > 0 else float("nan"),
                        f"y = {a_r} * ln(x) + {b_r}"
                    )

        # === Reciprocal: y = a/x + b ===
        if time.time() < deadline:
            nonzero_xs = [(x, y) for x, y in zip(xs, ys) if abs(x) > 1e-10]
            if len(nonzero_xs) >= 2:
                inv_xs = [1.0 / x for x, _ in nonzero_xs]
                ys_nz = [y for _, y in nonzero_xs]
                result = self._fit_linear_xy(inv_xs, ys_nz)
                if result:
                    a, b = result
                    a_r, b_r = round(a, precision), round(b, precision)
                    add_candidate(
                        lambda x, a=a_r, b=b_r: a / x + b if abs(x) > 1e-10 else float("nan"),
                        f"y = {a_r}/x + {b_r}"
                    )

        # === Square root: y = a * sqrt(x) + b ===
        if time.time() < deadline:
            nonneg_xs = [(x, y) for x, y in zip(xs, ys) if x >= 0]
            if len(nonneg_xs) >= 2:
                sqrt_xs = [math.sqrt(x) for x, _ in nonneg_xs]
                ys_nn = [y for _, y in nonneg_xs]
                result = self._fit_linear_xy(sqrt_xs, ys_nn)
                if result:
                    a, b = result
                    a_r, b_r = round(a, precision), round(b, precision)
                    add_candidate(
                        lambda x, a=a_r, b=b_r: a * math.sqrt(x) + b if x >= 0 else float("nan"),
                        f"y = {a_r} * sqrt(x) + {b_r}"
                    )

        # === Integer coefficient grid search (for exact relationships) ===
        if time.time() < deadline:
            remaining = deadline - time.time()
            if remaining > 0.2:
                # Try simple integer relationships
                for a_int in range(-10, 11):
                    if time.time() >= deadline:
                        break
                    for b_int in range(-10, 11):
                        # y = a*x + b
                        f = lambda x, a=a_int, b=b_int: a * x + b
                        err = score(f)
                        if err < 0.01:
                            add_candidate(f, f"y = {a_int}*x + {b_int} [exact]")

                        # y = x/a + b
                        if a_int != 0:
                            f2 = lambda x, a=a_int, b=b_int: x / a + b
                            err2 = score(f2)
                            if err2 < 0.01:
                                add_candidate(f2, f"y = x/{a_int} + {b_int} [exact]")

                        # y = (x + b) / a
                        if a_int != 0:
                            f3 = lambda x, a=a_int, b=b_int: (x + b) / a
                            err3 = score(f3)
                            if err3 < 0.01:
                                add_candidate(f3, f"y = (x + {b_int})/{a_int} [exact]")

        # === Floor/Ceil/Round variants ===
        if time.time() < deadline:
            # Test common rounding patterns
            for divisor in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                if time.time() >= deadline:
                    break
                for offset in range(-5, 6):
                    # y = floor((x + offset) / divisor)
                    f_floor = lambda x, d=divisor, o=offset: math.floor((x + o) / d)
                    if score(f_floor) < 0.01:
                        add_candidate(f_floor, f"y = floor((x + {offset})/{divisor})")

                    # y = round(x / divisor) + offset
                    f_round = lambda x, d=divisor, o=offset: round(x / d) + o
                    if score(f_round) < 0.01:
                        add_candidate(f_round, f"y = round(x/{divisor}) + {offset}")

                    # y = ceil((x + offset) / divisor)
                    f_ceil = lambda x, d=divisor, o=offset: math.ceil((x + o) / d)
                    if score(f_ceil) < 0.01:
                        add_candidate(f_ceil, f"y = ceil((x + {offset})/{divisor})")

        # === Quadratic with integer coefficients ===
        if time.time() < deadline and len(xs) >= 3:
            remaining = deadline - time.time()
            if remaining > 0.3:
                for a_int in range(-5, 6):
                    if time.time() >= deadline:
                        break
                    for b_int in range(-5, 6):
                        for c_int in range(-5, 6):
                            f = lambda x, a=a_int, b=b_int, c=c_int: a * x ** 2 + b * x + c
                            err = score(f)
                            if err < 0.01:
                                add_candidate(f, f"y = {a_int}*x² + {b_int}*x + {c_int} [exact]")

        # Sort by error
        candidates.sort(key=lambda t: t[1])
        return candidates

    def _forward_validate(
        self,
        f: Callable[[float], float],
        xs: list[float],
        ys: list[float],
        holdout: int = 2
    ) -> float:
        """
        Validate function by holding out the last `holdout` points.
        Returns mean absolute error on holdout set.
        """
        if len(xs) < holdout + 2:
            return float("inf")

        errors = []
        for i in range(len(xs) - holdout, len(xs)):
            try:
                pred = f(xs[i])
                if pred is not None and not math.isnan(pred) and not math.isinf(pred):
                    errors.append(abs(pred - ys[i]))
                else:
                    return float("inf")
            except Exception:
                return float("inf")

        return math.fsum(errors) / len(errors) if errors else float("inf")

    # ================================================================== #
    #  Regression Mode Prediction (Private)                               #
    # ================================================================== #

    def _predict_regression(
        self,
        reference: Dive,
        steps: int,
        TA: float
    ) -> list[float]:
        """
        Regression mode: reference is longer than self by `steps`.
        Use reference[0:n] aligned with self[0:n] to learn F(x) → y,
        then apply F to reference[n:n+steps] to predict self[n:n+steps].
        """
        n = len(self._data)
        x_history = reference._data[:n]
        x_future = reference._data[n:n + steps]
        y_history = self._data

        deadline = time.time() + TA

        # Phase 1: Discover mappings (60% of time)
        phase1_end = time.time() + TA * 0.6
        mappings = self._discover_mappings(x_history, y_history, phase1_end, precision=4)

        if not mappings:
            # Fallback: use correlation-based adjustment
            return self._predict_correlation_fallback(reference, steps)

        # Phase 2: Forward validation (20% of time)
        holdout = min(2, n // 4)
        validated: list[tuple[Callable, float, float, str]] = []

        for func, fit_err, desc in mappings:
            if time.time() >= deadline:
                break
            fwd_err = self._forward_validate(func, x_history, y_history, holdout)
            combined = fit_err + fwd_err * 2  # Weight forward validation higher
            validated.append((func, combined, fit_err, desc))

        validated.sort(key=lambda t: t[1])

        # Phase 3: Higher precision if time remains (15% of time)
        if time.time() < deadline and TA >= 3:
            phase3_end = time.time() + (deadline - time.time()) * 0.75
            hi_prec = self._discover_mappings(x_history, y_history, phase3_end, precision=6)
            for func, fit_err, desc in hi_prec[:10]:
                if time.time() >= deadline:
                    break
                fwd_err = self._forward_validate(func, x_history, y_history, holdout)
                combined = fit_err + fwd_err * 2
                validated.append((func, combined, fit_err, desc))
            validated.sort(key=lambda t: t[1])

        # Phase 4: Generate predictions
        if not validated:
            return self._predict_correlation_fallback(reference, steps)

        # Use best function or weighted ensemble of top functions
        best_funcs = validated[:3]  # Top 3 functions
        
        # Check if best is clearly dominant
        if len(best_funcs) >= 1 and best_funcs[0][1] < 0.01:
            # Near-perfect fit, use it directly
            func = best_funcs[0][0]
            results = []
            for x in x_future:
                pred = self._safe_call(func, x)
                if pred is not None:
                    results.append(self._round_if_close(pred))
                else:
                    results.append(y_history[-1])  # Fallback
            return results

        # Weighted ensemble of top functions
        results = []
        for x in x_future:
            weighted_sum = 0.0
            weight_total = 0.0
            for func, combined_err, _, _ in best_funcs:
                pred = self._safe_call(func, x)
                if pred is not None:
                    weight = 1.0 / (combined_err + 1e-10)
                    weighted_sum += weight * pred
                    weight_total += weight
            if weight_total > 0:
                results.append(self._round_if_close(weighted_sum / weight_total))
            else:
                results.append(y_history[-1])

        return results

    def _predict_correlation_fallback(self, reference: Dive, steps: int) -> list[float]:
        """Fallback when function mapping fails: use correlation-based adjustment."""
        n = len(self._data)
        base_pred = self._ensemble_predict(steps)

        # Get reference's prediction for its extra points
        ref_future = reference._data[n:n + steps]

        # Compute correlation on aligned portion
        try:
            aligned_ref = Dive(reference._data[:n])
            corr = self.correlation(aligned_ref)
        except Exception:
            return base_pred

        if abs(corr) < 0.1:
            return base_pred

        # Adjust based on how reference's future deviates from its mean
        ref_mean = aligned_ref.mean()
        ref_std = aligned_ref.stdev() if n >= 2 else 1.0
        self_std = self.stdev() if n >= 2 else 1.0

        if ref_std < 1e-10:
            return base_pred

        adjustments = [
            (x - ref_mean) * (self_std / ref_std) * corr
            for x in ref_future
        ]

        return [bp + adj for bp, adj in zip(base_pred, adjustments)]

    # ================================================================== #
    #  Correlation Mode Prediction (Private)                              #
    # ================================================================== #

    def _predict_with_correlation(
        self,
        reference: Dive,
        steps: int,
        corr_threshold: float
    ) -> list[float]:
        """
        Correlation mode: reference has same length as self.
        Use correlation to adjust time-series predictions.
        """
        n = len(self._data)

        # Get base prediction from time-series methods
        base_pred = self._time_series_predict(steps)

        # Get reference's prediction
        ref_pred = reference._ensemble_predict(steps)

        # Compute correlation
        try:
            corr = self.correlation(reference)
        except Exception:
            return base_pred

        if abs(corr) < corr_threshold:
            return base_pred

        # Adjust based on correlation
        self_std = self.stdev() if n >= 2 else 1.0
        ref_std = reference.stdev() if n >= 2 else 1.0

        if ref_std < 1e-10:
            return base_pred

        ref_mean = reference.mean()
        adjustments = [
            (rp - ref_mean) * (self_std / ref_std) * corr * 0.5
            for rp in ref_pred
        ]

        return [bp + adj for bp, adj in zip(base_pred, adjustments)]

    # ================================================================== #
    #  Time-Series Only Prediction (Private)                              #
    # ================================================================== #

    def _time_series_predict(self, steps: int) -> list[float]:
        """
        Pure time-series prediction using exact detection + ensemble fallback.
        """
        n = len(self._data)
    
        # Try exact polynomial
        poly_pred = self._predict_polynomial_exact(steps)
        if poly_pred is not None:
            order = self._detect_polynomial_order()
            min_sub = max(3, (order or 0) + 2)
            if self._verify_pattern("_predict_polynomial_exact", min_sub, steps):
                return poly_pred
    
        # Try exact geometric
        geo_pred = self._predict_geometric(steps)
        if geo_pred is not None:
            if self._verify_pattern("_predict_geometric", 3, steps):
                return geo_pred
    
        # Try ratio pattern
        ratio_pred = self._predict_ratio_pattern(steps)
        if ratio_pred is not None:
            if self._verify_pattern("_predict_ratio_pattern", 4, steps):
                return ratio_pred
    
        # Fallback to ensemble
        return self._ensemble_predict(steps)

    def _verify_pattern(
        self,
        method_name: str,
        min_points: int,
        steps: int
    ) -> bool:
        """Verify a detected pattern by backtesting on shorter subsequences."""
        n = len(self._data)
        if n < min_points + 1:
            return True  # Trust detection for small datasets
    
        checks_done = 0
        checks_passed = 0
    
        for offset in range(1, min(4, n - min_points + 1)):
            sub = Dive(self._data[:n - offset])
            if len(sub._data) < min_points:
                break
            
            try:
                method = getattr(sub, method_name)
                sub_pred = method(1)
            except Exception:
                sub_pred = None
    
            if sub_pred is None:
                break
            checks_done += 1
            if abs(sub_pred[0] - self._data[n - offset]) < 1e-6:
                checks_passed += 1
    
        return checks_done == 0 or checks_passed == checks_done

    # ================================================================== #
    #  Public Prediction API                                              #
    # ================================================================== #

    def predict_next(
        self,
        steps: int = 1,
        *,
        method: str = "ensemble",
        reference: Dive | None = None,
        corr_threshold: float = 0.1,
        TA: float = 0,
    ) -> float | list[float]:
        """
        Predict the next value(s) in the series.

        Parameters
        ----------
        steps : int
            Number of future values to predict (default 1).
        method : str
            Prediction method. Options:
            - "ensemble" (default): weighted combination of all models
            - "linear", "quadratic", "holt", "exponential", "differences",
              "drift", "newton", "lagrange"
        reference : Dive | None
            Optional reference series for enhanced prediction.
            - If len(reference) == len(self) + steps: **Regression mode**
              Uses reference as predictor variable. The extra elements in
              reference are known future values used to predict self.
            - If len(reference) == len(self): **Correlation mode**
              Adjusts predictions based on correlation between series.
        corr_threshold : float
            Minimum correlation for reference adjustment (default 0.1).
        TA : float
            Time Allotment in seconds (default 0). When > 0 with a
            reference, enables deep analysis including function mapping
            discovery, forward validation, and extensive search for the
            optimal F(reference) → self relationship.

        Returns
        -------
        float or list[float]
            Single value if steps=1, otherwise list of predictions.

        Examples
        --------
        >>> sales = Dive([100, 150, 120, 200, 180])
        >>> temps = Dive([20, 25, 22, 30, 28, 35])  # 35 is tomorrow's temp
        >>> sales.predict_next(reference=temps, TA=1)
        220.5  # predicted sales for temperature 35
        """
        self._require(2)
        n = len(self._data)

        # Determine mode based on reference length
        if reference is not None:
            ref_len = len(reference._data)

            if ref_len == n + steps:
                # REGRESSION MODE: reference contains known future predictors
                if TA > 0:
                    results = self._predict_regression(reference, steps, TA)
                else:
                    # Quick regression without TA
                    results = self._predict_regression(reference, steps, 0.5)

                if steps == 1:
                    return results[0]
                return results

            elif ref_len == n:
                # CORRELATION MODE: parallel series adjustment
                if method == "ensemble":
                    results = self._predict_with_correlation(reference, steps, corr_threshold)
                    if steps == 1:
                        return results[0]
                    return results
            else:
                raise ValueError(
                    f"Reference length ({ref_len}) must equal self length ({n}) "
                    f"for correlation mode, or self length + steps ({n + steps}) "
                    f"for regression mode."
                )

        # NO REFERENCE: pure time-series prediction
        if method == "ensemble":
            results = self._time_series_predict(steps)
            if steps == 1:
                return results[0]
            return results

        # Specific method requested
        if method == "linear":
            preds = [self._predict_linear(n - 1 + s) for s in range(1, steps + 1)]
        elif method == "quadratic":
            preds = [self._predict_quadratic(n - 1 + s) for s in range(1, steps + 1)]
        elif method == "holt":
            preds = self._predict_holt(steps)
        elif method == "exponential":
            preds = []
            for s in range(1, steps + 1):
                ep = self._predict_exponential(n - 1 + s)
                preds.append(ep if ep is not None else self._data[-1])
        elif method == "differences":
            preds = self._predict_differences(steps)
        elif method == "drift":
            preds = self._predict_drift(steps)
        elif method == "newton":
            preds = self._predict_newton(steps)
        elif method == "lagrange":
            preds = [self._predict_lagrange(n - 1 + s) for s in range(1, steps + 1)]
        elif method == "seasonal":
            preds = self._predict_seasonal_naive(steps)
        else:
            raise ValueError(f"Unknown method '{method}'.")

        if steps == 1:
            return preds[0]
        return preds

    def predict_detail(self, steps: int = 1, reference: Dive | None = None, TA: float = 0) -> dict:
        """
        Return detailed prediction information from all models.

        Parameters
        ----------
        steps : int
            Number of future values to predict.
        reference : Dive | None
            Optional reference series (see predict_next for modes).
        TA : float
            Time allotment for deep analysis.

        Returns
        -------
        dict
            Contains predictions from each model, backtest errors, weights,
            and discovered function mappings (if reference provided with TA).
        """
        self._require(2)
        n = len(self._data)
        k = min(max(3, n // 3), 10)
        maes = self._backtest(k)
        weights = self._inverse_weights(maes)
        preds = self._get_all_predictions(steps)

        result = {
            "predictions": preds,
            "backtest_mae": maes,
            "weights": weights,
            "ensemble": self._ensemble_predict(steps),
            "final": self.predict_next(steps, reference=reference, TA=TA),
        }

        # Add exact detection results
        poly_exact = self._predict_polynomial_exact(steps)
        if poly_exact:
            result["polynomial_exact"] = poly_exact
            result["polynomial_order"] = self._detect_polynomial_order()

        geo_exact = self._predict_geometric(steps)
        if geo_exact:
            result["geometric_exact"] = geo_exact
            result["geometric_ratio"] = self._detect_geometric()

        ratio_exact = self._predict_ratio_pattern(steps)
        if ratio_exact:
            result["ratio_pattern"] = ratio_exact

        # Add function mappings if reference provided
        if reference is not None and TA > 0:
            ref_len = len(reference._data)
            if ref_len == n + steps:
                x_history = reference._data[:n]
                y_history = self._data
                mappings = self._discover_mappings(x_history, y_history, time.time() + TA * 0.5, precision=4)
                result["discovered_mappings"] = [
                    {"function": desc, "fit_error": err}
                    for _, err, desc in mappings[:10]
                ]

        return result

    # ================================================================== #
    #  Transforms                                                         #
    # ================================================================== #

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

    def diff(self, periods: int = 1) -> Dive:
        """Return first differences (or n-th differences if periods > 1)."""
        if periods < 1:
            raise ValueError("periods must be >= 1.")
        result = list(self._data)
        for _ in range(periods):
            result = [result[i + 1] - result[i] for i in range(len(result) - 1)]
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

    def clip(self, lo: float | None = None, hi: float | None = None) -> Dive:
        """Clip values to [lo, hi] range."""
        result = []
        for x in self._data:
            if lo is not None and x < lo:
                x = lo
            if hi is not None and x > hi:
                x = hi
            result.append(x)
        return Dive(result)

    def apply(self, func: Callable[[float], float]) -> Dive:
        """Apply a function to each element."""
        return Dive([func(x) for x in self._data])

    # ================================================================== #
    #  Outlier Detection                                                  #
    # ================================================================== #

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

    # ================================================================== #
    #  Cross-Dataset                                                      #
    # ================================================================== #

    def correlation(self, other: Dive, *, outliers: float = 0) -> float:
        if not isinstance(other, Dive):
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

    def covariance(self, other: Dive) -> float:
        if not isinstance(other, Dive):
            raise TypeError("covariance() requires another Dive instance.")
        n = len(self._data)
        if n != len(other._data):
            raise ValueError("Both datasets must have the same length.")
        self._require(2)
        mx, my = self.mean(), _stats.mean(other._data)
        return math.fsum(
            (x - mx) * (y - my) for x, y in zip(self._data, other._data)
        ) / (n - 1)

    def regress_on(self, other: Dive) -> tuple[float, float, float]:
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

    # ================================================================== #
    #  Summary & Visualisation                                            #
    # ================================================================== #

    def summary(self) -> str:
        self._require()
        n = len(self._data)
        rows: list[tuple[str, str]] = [
            ("Count", f"{n}"),
            ("Sum", f"{self.sum():.6g}"),
            ("Mean", f"{self.mean():.6g}"),
            ("Median", f"{self.median():.6g}"),
            ("Mode(s)", ", ".join(f"{m:g}" for m in self.mode())),
            ("Std Dev", f"{self.stdev():.6g}" if n >= 2 else "N/A"),
            ("Variance", f"{self.variance():.6g}" if n >= 2 else "N/A"),
            ("Min", f"{self.min():.6g}"),
            ("Max", f"{self.max():.6g}"),
            ("Range", f"{self.range():.6g}"),
            ("Q1", f"{self.percentile(25):.6g}"),
            ("Q3", f"{self.percentile(75):.6g}"),
            ("IQR", f"{self.iqr():.6g}"),
        ]
        if n >= 3:
            rows.append(("Skewness", f"{self.skewness():.6g}"))
        if n >= 4:
            rows.append(("Kurtosis", f"{self.kurtosis():.6g}"))

        lw = max(len(lbl) for lbl, _ in rows)
        vw = max(len(val) for _, val in rows)
        w = lw + vw + 5
        hr = "-" * w
        title = " DIVE Summary "
        pad = w - len(title)
        header = "-" * (pad // 2) + title + "-" * (pad - pad // 2)
        lines = [f"  {header}"]
        for lbl, val in rows:
            lines.append(f"  | {lbl:<{lw}}  {val:>{vw}} |")
        lines.append(f"  {hr}")
        return "\n".join(lines)

    describe = summary

    def histogram(self, bins: int = 10, *, width: int = 40, char: str = "#") -> str:
        self._require()
        s = sorted(self._data)
        lo, hi = s[0], s[-1]
        if lo == hi:
            return f"\n  {lo:.4g} | {char * width}  ({len(self._data)})\n"
        bin_w = (hi - lo) / bins
        counts = [0] * bins
        for x in self._data:
            idx = min(int((x - lo) / bin_w), bins - 1)
            counts[idx] += 1
        mx = max(counts)
        lines: list[str] = ["", "  Histogram", "  " + "-" * (width + 30)]
        for i, c in enumerate(counts):
            edge_l = lo + i * bin_w
            edge_r = edge_l + bin_w
            bar_len = round((c / mx) * width) if mx else 0
            lines.append(
                f"  {edge_l:>9.4g} - {edge_r:<9.4g} | "
                f"{char * bar_len:<{width}}  ({c})"
            )
        lines.append("")
        return "\n".join(lines)

    def sparkline(self) -> str:
        if not self._data:
            return ""
        blocks = " ▁▂▃▄▅▆▇█"
        lo, hi = self.min(), self.max()
        r = hi - lo or 1.0
        return "".join(
            blocks[min(int((v - lo) / r * 8), 8)] for v in self._data
        )

    def plot_ascii(self, width: int = 60, height: int = 15) -> str:
        """Simple ASCII line plot."""
        self._require()
        n = len(self._data)
        lo, hi = self.min(), self.max()
        r = hi - lo or 1.0

        # Create canvas
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, v in enumerate(self._data):
            x = int((i / (n - 1)) * (width - 1)) if n > 1 else 0
            y = int(((v - lo) / r) * (height - 1))
            y = height - 1 - y  # Flip y-axis
            canvas[y][x] = "●"

        # Connect points
        for i in range(n - 1):
            x1 = int((i / (n - 1)) * (width - 1))
            x2 = int(((i + 1) / (n - 1)) * (width - 1))
            y1 = int(((self._data[i] - lo) / r) * (height - 1))
            y2 = int(((self._data[i + 1] - lo) / r) * (height - 1))
            y1 = height - 1 - y1
            y2 = height - 1 - y2

            for x in range(x1, x2):
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                y = int(y1 + t * (y2 - y1))
                if canvas[y][x] == " ":
                    canvas[y][x] = "·"

        lines = ["", f"  max: {hi:.4g}"]
        for row in canvas:
            lines.append("  │" + "".join(row) + "│")
        lines.append(f"  min: {lo:.4g}")
        lines.append(f"  └{'─' * width}┘")
        lines.append(f"   0{' ' * (width - 5)}{n - 1}")
        return "\n".join(lines)

    # ================================================================== #
    #  Export                                                              #
    # ================================================================== #

    def to_list(self) -> list[float]:
        return list(self._data)

    def to_dict(self) -> dict:
        self._require()
        n = len(self._data)
        d: dict = {
            "count": n,
            "sum": self.sum(),
            "mean": self.mean(),
            "median": self.median(),
            "mode": self.mode(),
            "min": self.min(),
            "max": self.max(),
            "range": self.range(),
            "q1": self.percentile(25),
            "q3": self.percentile(75),
            "iqr": self.iqr(),
        }
        if n >= 2:
            d["stdev"] = self.stdev()
            d["variance"] = self.variance()
        return d


# ====================================================================== #
#  Quick Test                                                             #
# ====================================================================== #

if __name__ == "__main__":
    all_data={}
    main=Dive()

    def F1(x):
        return x**2
    def F2(x):
        return x*2
    def F3(x):
        return x+1
    def F4(x):
        return x-1
    def F5(x):
        return x**3
    def F6(x):
        return 2**x
    def F7(x):
        return x**2 + 3*x - 7
    def F8(x):
        return x*(x+1)//2
    def F9(x):
        return x**4
    def F10(x):
        return 3*x**3 - 2*x**2 + x - 5


    for func in [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]:
        for a in range(6,50):
            goal=func(a-2)
            d1=Dive([func(i) for i in range(1,a-1)])
            d2=Dive([func(i) for i in range(2,a)])

            pred=abs(goal-d1.predict_next())
            predwr=abs(goal-d1.predict_next(reference=d2,TA=1))
            
            main.add(pred)
            main.add(predwr)

            if pred!=0 or predwr!=0:
                all_data[str(func)+str(a)]=pred
                all_data[str(func)+str(a)+'R']=predwr

    print('averages:')
    print(main.mean())
    print(main.geo_mean(filter_bad=True))#    filter_bad=True replaces sub-zero values with 0.0001
    print(main.harmonic_mean(filter_bad=True))
    print('\nmin/max:')
    print(main.min())
    print(main.max())
    print('\nmost common/middle value:')