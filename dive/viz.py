"""
DIVE Visualization — Summary and plotting methods.
"""

from .stats import DiveStats


class DiveViz(DiveStats):
    """Visualization methods for Dive."""

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