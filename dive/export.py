"""
DIVE Export — Data export methods.
"""

class DiveExport:
    """Export methods for Dive."""

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