"""
DIVE Stats — Statistical analysis methods.
"""

import math
import statistics as _stats
from typing import Any


from .base import DiveBase


class DiveStats(DiveBase):
    """Statistical methods for Dive."""

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
        data = self._data
        if filter_bad:
            data = [x for x in data if x > 0]
            if not data:
                raise ValueError("No positive values found in dataset after filtering.")
        else:
            if any(x <= 0 for x in data):
                raise ValueError("Geometric mean requires all strictly positive values.")
        
        return math.exp(
            math.fsum(math.log(x) for x in data) / len(data)
        )

    def harmonic_mean(self, filter_bad: bool = False) -> float:
        self._require()
        data = self._data
        if filter_bad:
            data = [x for x in data if x > 0]
            if not data:
                raise ValueError("No positive values found in dataset after filtering.")
        else:
            if any(x <= 0 for x in data):
                raise ValueError("Harmonic mean requires all strictly positive values.")
        return _stats.harmonic_mean(data)

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