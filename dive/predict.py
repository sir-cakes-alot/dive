"""
DIVE Predict — Time-series prediction and analysis methods.
"""

import math
import time
from typing import Any, Callable


from .transforms import DiveTransforms


class DivePredict(DiveTransforms):
    """Prediction methods for Dive."""

    # Time-Series Pattern Detection

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

    # Time-Series Prediction Models

    def _is_finite_value(self, value: float) -> bool:
        return isinstance(value, (float, int)) and not (math.isnan(value) or math.isinf(value))

    def _bound_prediction(self, value: float) -> float:
        if not self._is_finite_value(value):
            raise ValueError("Non-finite prediction")
        if not self._data:
            return value
        last_val = self._data[-1]
        stdev_val = self.stdev(population=True) if len(self._data) > 1 else 0.0
        max_delta = max(abs(last_val) * 10.0, stdev_val * 20.0, 1.0)
        if value > last_val + max_delta:
            return last_val + max_delta
        if value < last_val - max_delta:
            return last_val - max_delta
        return value

    def _predict_lagrange(self, index: float) -> float:
        n = len(self._data)
        if n == 0:
            return 0.0
        if n == 1:
            return self._data[0]

        window = min(n, 8)
        start = n - window
        window_data = self._data[start:]

        result = 0.0
        for i in range(window):
            xi = start + i
            term = window_data[i]
            denom = 1.0
            for j in range(window):
                if i == j:
                    continue
                xj = start + j
                denom *= (xi - xj)
                term *= (index - xj)
            result += term / denom

        bounded = self._bound_prediction(self._round_if_close(result))
        return self._round_if_close(bounded)

    def _predict_newton(self, steps: int = 1) -> list[float]:
        n = len(self._data)
        if n == 0:
            return [0.0] * steps
        if n == 1:
            return [self._data[0]] * steps

        window = min(n, 8)
        base = self._data[-window:]

        table: list[list[float]] = [list(base)]
        for k in range(1, window):
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

        max_keep = min(4, len(tails))
        keep = max(2, min(effective_order + 1, max_keep))
        tails = tails[:keep]

        results = []
        for _ in range(steps):
            for i in range(len(tails) - 2, -1, -1):
                tails[i] = tails[i] + tails[i + 1]
            candidate = self._round_if_close(tails[0])
            candidate = self._bound_prediction(candidate)
            results.append(self._round_if_close(candidate))

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
                return 0.0, 0.0, 0.0
            for row in range(col + 1, 3):
                factor = mat[row][col] / mat[col][col]
                for c in range(col, 4):
                    mat[row][c] -= factor * mat[col][c]

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
                xtx[j][k] = math.fsum(powers[i][j] * powers[i][k] for i in range(n))
            xty[j] = math.fsum(powers[i][j] * self._data[i] for i in range(n))

        aug = [xtx[r][:] + [xty[r]] for r in range(d)]
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-15:
                return [0.0] * d
            for row in range(col + 1, d):
                factor = aug[row][col] / aug[col][col]
                for c in range(col, d + 1):
                    aug[row][c] -= factor * aug[col][c]

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
                mse = 0.0
                for t in range(1, len(self._data)):
                    new_level = a * self._data[t] + (1 - a) * (level + trend)
                    new_trend = b * (new_level - level) + (1 - b) * trend
                    level, trend = new_level, new_trend
                    if t >= 2:
                        pred = level + trend
                        mse += (self._data[t] - pred) ** 2
                if mse < best_mse:
                    best, best_mse = (a, b), mse

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

    # Ensemble Prediction Engine

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
            sub = type(self)(self._data[:split])
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
                    errors[name].append(abs(pred - actual))

            pred = self._safe_call(lambda s=sub, sp=split: s._predict_exponential(sp))
            if pred is not None:
                errors["exponential"].append(abs(pred - actual))

            pred = self._safe_call(lambda s=sub: s._predict_seasonal_naive(1)[0])
            if pred is not None:
                errors["seasonal"].append(abs(pred - actual))

            poly_order = sub._detect_polynomial_order()
            deg = poly_order if poly_order is not None else min(3, split - 1)
            pred = self._safe_call(lambda s=sub, d=deg, sp=split: s._predict_polynomial(d, sp))
            if pred is not None:
                errors["poly_auto"].append(abs(pred - actual))

            for deg in [3, 4, 5]:
                if split > deg:
                    pred = self._safe_call(lambda s=sub, d=deg, sp=split: s._predict_polynomial(d, sp))
                    if pred is not None:
                        errors[f"poly_{deg}"].append(abs(pred - actual))

            if split >= 3:
                pred = self._safe_call(lambda s=sub, sp=split: s._predict_quadratic(sp))
                if pred is not None:
                    errors["quadratic"].append(abs(pred - actual))
                pred = self._safe_call(lambda s=sub: s._predict_holt(1)[0])
                if pred is not None:
                    errors["holt"].append(abs(pred - actual))

        return {m: (math.fsum(e) / len(e) if e else float("inf")) for m, e in errors.items()}

    @staticmethod
    def _inverse_weights(maes: dict[str, float]) -> dict[str, float]:
        eps = 1e-15
        max_reasonable_mae = 1e8
        finite = {m: v for m, v in maes.items() if v < float("inf") and v <= max_reasonable_mae}
        if not finite:
            return {m: 1.0 / len(maes) for m in maes}

        min_mae = min(finite.values())
        if min_mae < 1e-12:
            return {m: 1.0 if v == min_mae else 0.0 for m, v in finite.items()}

        inv = {m: 1.0 / (v + eps) for m, v in finite.items()}
        total = sum(inv.values())
        normalized = {m: inv.get(m, 0.0) / total for m in maes}

        # Clip extreme weights and renormalize
        clipped = {m: min(max(w, 0.0), 0.5) for m, w in normalized.items()}
        clip_total = sum(clipped.values())
        if clip_total > 0:
            return {m: w / clip_total for m, w in clipped.items()}
        return {m: 1.0 / len(maes) for m in maes}

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

        # remove clearly unstable models from ensemble
        stable_preds = {}
        for model, pred_list in preds.items():
            if any(not self._is_finite_value(p) for p in pred_list):
                continue
            if not self._data:
                stable_preds[model] = pred_list
                continue
            last_val = self._data[-1]
            max_delta = max(abs(last_val) * 20.0, self.stdev(population=True) * 40.0, 1.0)
            if any(abs(p - last_val) > max_delta for p in pred_list):
                continue
            stable_preds[model] = pred_list

        if not stable_preds:
            stable_preds = preds

        results = []
        for step_idx in range(steps):
            weighted_sum = 0.0
            weight_total = 0.0
            for model, pred_list in stable_preds.items():
                if step_idx < len(pred_list):
                    w = weights.get(model, 0.0)
                    weighted_sum += w * pred_list[step_idx]
                    weight_total += w
            if weight_total > 0:
                candidate = weighted_sum / weight_total
            else:
                candidate = self._data[-1] if self._data else 0.0

            results.append(self._bound_prediction(candidate))

        return results

    # Function Mapping Discovery

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
        slope = num / den
        intercept = y_mean - slope * x_mean
        return slope, intercept

    @staticmethod
    def _fit_polynomial_xy(xs: list[float], ys: list[float], degree: int) -> list[float] | None:
        n = len(xs)
        if n < degree + 1 or n != len(ys):
            return None
        d = degree + 1
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d

        powers = []
        for i in range(n):
            row = [1.0]
            for _ in range(1, 2 * degree + 1):
                row.append(row[-1] * xs[i])
            powers.append(row)

        for j in range(d):
            for k in range(d):
                xtx[j][k] = math.fsum(powers[i][j] * powers[i][k] for i in range(n))
            xty[j] = math.fsum(powers[i][j] * ys[i] for i in range(n))

        aug = [xtx[r][:] + [xty[r]] for r in range(d)]
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-15:
                return None
            for row in range(col + 1, d):
                factor = aug[row][col] / aug[col][col]
                for c in range(col, d + 1):
                    aug[row][c] -= factor * aug[col][c]

        coeffs = [0.0] * d
        for i in range(d - 1, -1, -1):
            coeffs[i] = aug[i][d]
            for j in range(i + 1, d):
                coeffs[i] -= aug[i][j] * coeffs[j]
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
            errors = []
            for x, y in zip(xs, ys):
                try:
                    pred = f(x)
                    if math.isfinite(pred):
                        errors.append(abs(pred - y))
                except:
                    pass
            return math.fsum(errors) / len(errors) if errors else float("inf")

        def add_candidate(f: Callable, desc: str) -> None:
            err = score(f)
            if err < float("inf"):
                candidates.append((f, err, desc))

        # === Linear: y = a*x + b ===
        if time.time() < deadline:
            params = self._fit_linear_xy(xs, ys)
            if params:
                a, b = params
                add_candidate(lambda x: a * x + b, f"Linear: {a:.{precision}g}*x + {b:.{precision}g}")

        # === Polynomial degrees 2-6 ===
        for deg in range(2, 7):
            if time.time() < deadline:
                coeffs = self._fit_polynomial_xy(xs, ys, deg)
                if coeffs:
                    desc_parts = [f"{coeffs[0]:.{precision}g}"]
                    for i, c in enumerate(coeffs[1:], 1):
                        if c >= 0:
                            desc_parts.append(f"+ {c:.{precision}g}*x^{i}")
                        else:
                            desc_parts.append(f"- {-c:.{precision}g}*x^{i}")
                    desc = "Polynomial: " + " ".join(desc_parts)
                    add_candidate(lambda x, cs=coeffs: sum(c * (x ** i) for i, c in enumerate(cs)), desc)

        # === Power: y = a * x^b ===
        if time.time() < deadline:
            valid = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                log_x = [math.log(x) for x in vx]
                log_y = [math.log(y) for y in vy]
                params = self._fit_linear_xy(log_x, log_y)
                if params:
                    a, b = params
                    a = math.exp(a)
                    add_candidate(lambda x: a * (x ** b), f"Power: {a:.{precision}g} * x^{b:.{precision}g}")

        # === Exponential: y = a * e^(b*x) ===
        if time.time() < deadline:
            valid = [(x, y) for x, y in zip(xs, ys) if y > 0]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                log_y = [math.log(y) for y in vy]
                params = self._fit_linear_xy(vx, log_y)
                if params:
                    b, log_a = params
                    a = math.exp(log_a)
                    add_candidate(lambda x: a * math.exp(b * x), f"Exponential: {a:.{precision}g} * e^({b:.{precision}g}*x)")

        # === Logarithmic: y = a * ln(x) + b ===
        if time.time() < deadline:
            valid = [(x, y) for x, y in zip(xs, ys) if x > 0]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                log_x = [math.log(x) for x in vx]
                params = self._fit_linear_xy(log_x, vy)
                if params:
                    a, b = params
                    add_candidate(lambda x: a * math.log(x) + b, f"Logarithmic: {a:.{precision}g} * ln(x) + {b:.{precision}g}")

        # === Reciprocal: y = a/x + b ===
        if time.time() < deadline:
            valid = [(x, y) for x, y in zip(xs, ys) if x != 0]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                rec_x = [1 / x for x in vx]
                params = self._fit_linear_xy(rec_x, vy)
                if params:
                    a, b = params
                    add_candidate(lambda x: a / x + b, f"Reciprocal: {a:.{precision}g}/x + {b:.{precision}g}")

        # === Square root: y = a * sqrt(x) + b ===
        if time.time() < deadline:
            valid = [(x, y) for x, y in zip(xs, ys) if x >= 0]
            if len(valid) >= 2:
                vx, vy = zip(*valid)
                sqrt_x = [math.sqrt(x) for x in vx]
                params = self._fit_linear_xy(sqrt_x, vy)
                if params:
                    a, b = params
                    add_candidate(lambda x: a * math.sqrt(x) + b, f"Square root: {a:.{precision}g} * sqrt(x) + {b:.{precision}g}")

        # === Integer coefficient grid search (for exact relationships) ===
        if time.time() < deadline:
            # Simple grid search for y = a*x + b with integer a, b
            for a in range(-10, 11):
                for b in range(-10, 11):
                    if a == 0 and b == 0:
                        continue
                    err = score(lambda x: a * x + b)
                    if err < 0.1:  # very close fit
                        add_candidate(lambda x, aa=a, bb=b: aa * x + bb, f"Integer linear: {aa}*x + {bb}")

        # === Floor/Ceil/Round variants ===
        if time.time() < deadline:
            for func_name, func in [("floor", math.floor), ("ceil", math.ceil), ("round", round)]:
                err = score(func)
                if err < 0.1:
                    add_candidate(func, f"{func_name.capitalize()}(x)")

        # === Quadratic with integer coefficients ===
        if time.time() < deadline and len(xs) >= 3:
            for a in range(-5, 6):
                for b in range(-10, 11):
                    for c in range(-10, 11):
                        if a == 0 and b == 0 and c == 0:
                            continue
                        err = score(lambda x, aa=a, bb=b, cc=c: aa * x**2 + bb * x + cc)
                        if err < 0.1:
                            desc = f"Integer quadratic: {aa}*x^2"
                            if bb >= 0:
                                desc += f" + {bb}*x"
                            else:
                                desc += f" - {-bb}*x"
                            if cc >= 0:
                                desc += f" + {cc}"
                            else:
                                desc += f" - {-cc}"
                            add_candidate(lambda x, aa=a, bb=b, cc=c: aa * x**2 + bb * x + cc, desc)

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
                if math.isfinite(pred):
                    errors.append(abs(pred - ys[i]))
            except:
                pass

        return math.fsum(errors) / len(errors) if errors else float("inf")

    # Regression Mode Prediction

    def _predict_regression(
        self,
        reference: type(self),
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
            return self._predict_correlation_fallback(reference, steps)

        # Phase 2: Forward validation (20% of time)
        holdout = min(2, n // 4)
        validated: list[tuple[Callable, float, float, str]] = []

        for func, fit_err, desc in mappings:
            val_err = self._forward_validate(func, x_history, y_history, holdout)
            if val_err < float("inf"):
                validated.append((func, fit_err, val_err, desc))

        validated.sort(key=lambda t: t[2])

        # Phase 4: Generate predictions
        if not validated:
            return self._predict_correlation_fallback(reference, steps)

        # Use best function or weighted ensemble of top functions
        best_funcs = validated[:3]  # Top 3 functions
        
        # Check if best is clearly dominant
        if len(best_funcs) >= 1 and best_funcs[0][2] < 0.01:
            func = best_funcs[0][0]
            return [func(x) for x in x_future]

        # Weighted ensemble of top functions
        results = []
        for x in x_future:
            preds = []
            weights = []
            for func, fit_err, val_err, _ in best_funcs:
                try:
                    pred = func(x)
                    if math.isfinite(pred):
                        preds.append(pred)
                        weights.append(1.0 / (val_err + 1e-6))
                except:
                    pass
            if preds:
                total_weight = sum(weights)
                weighted_pred = sum(p * w for p, w in zip(preds, weights)) / total_weight
                results.append(weighted_pred)
            else:
                results.append(self._data[-1])
        return results

    def _predict_correlation_fallback(self, reference: type(self), steps: int) -> list[float]:
        """Fallback when function mapping fails: use correlation-based adjustment."""
        n = len(self._data)
        base_pred = self._ensemble_predict(steps)

        # Get reference's prediction for its extra points
        ref_future = reference._data[n:n + steps]

        # Compute correlation on aligned portion
        try:
            aligned_ref = type(self)(reference._data[:n])
            corr = self.correlation(aligned_ref)
        except Exception:
            corr = 0.0

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

    # Correlation Mode Prediction

    def _predict_with_correlation(
        self,
        reference: type(self),
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
            corr = 0.0

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

    # Time-Series Only Prediction

    def _time_series_predict(self, steps: int) -> list[float]:
        """
        Pure time-series prediction using exact detection + ensemble fallback.
        """
        n = len(self._data)
    
        # Try exact polynomial
        poly_pred = self._predict_polynomial_exact(steps)
        if poly_pred is not None:
            return poly_pred
    
        # Try exact geometric
        geo_pred = self._predict_geometric(steps)
        if geo_pred is not None:
            return geo_pred
    
        # Try ratio pattern
        ratio_pred = self._predict_ratio_pattern(steps)
        if ratio_pred is not None:
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
            return False
    
        checks_done = 0
        checks_passed = 0
    
        for offset in range(1, min(4, n - min_points + 1)):
            sub = type(self)(self._data[:-offset])
            if len(sub._data) < min_points:
                continue
            checks_done += 1
            # Simplified verification
            try:
                pred = getattr(sub, f"_predict_{method_name}")(1)[0]
                actual = self._data[-offset]
                if abs(pred - actual) < 0.1:
                    checks_passed += 1
            except:
                pass
    
        return checks_done == 0 or checks_passed == checks_done

    # Public Prediction API

    def predict_next(
        self,
        steps: int = 1,
        *,
        method: str = "ensemble",
        reference: type(self) | None = None,
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
        >>> temps = Dive([20, 25, 22, 30, 28, 35])  # reference has steps=1 extra values
        >>> sales.predict_next(reference=temps, TA=1)
        220.5  # predicted sales for temperature 35
        """
        self._require(2)
        n = len(self._data)

        # Determine mode based on reference length
        if reference is not None:
            ref_len = len(reference._data)
            if ref_len == n + steps:
                # Regression mode
                preds = self._predict_regression(reference, steps, TA)
            elif ref_len == n:
                # Correlation mode
                preds = self._predict_with_correlation(reference, steps, corr_threshold)
            else:
                raise ValueError(
                    f"Reference length {ref_len} incompatible with self length {n} "
                    f"and steps {steps}. Expected {n + steps} (regression) or {n} (correlation)."
                )
        else:
            # NO REFERENCE: pure time-series prediction
            if method == "ensemble":
                preds = self._ensemble_predict(steps)
            else:
                preds = self._get_specific_prediction(method, steps)

        if steps == 1:
            return preds[0]
        return preds

    def _get_specific_prediction(self, method: str, steps: int) -> list[float]:
        if method == "linear":
            return [self._predict_linear(len(self._data) - 1 + i) for i in range(1, steps + 1)]
        elif method == "quadratic":
            return [self._predict_quadratic(len(self._data) - 1 + i) for i in range(1, steps + 1)]
        elif method == "holt":
            return self._predict_holt(steps)
        elif method == "exponential":
            preds = []
            for i in range(1, steps + 1):
                pred = self._predict_exponential(len(self._data) - 1 + i)
                preds.append(pred if pred is not None else self._data[-1])
            return preds
        elif method == "differences":
            return self._predict_differences(steps)
        elif method == "drift":
            return self._predict_drift(steps)
        elif method == "newton":
            return self._predict_newton(steps)
        elif method == "lagrange":
            return [self._predict_lagrange(len(self._data) - 1 + i) for i in range(1, steps + 1)]
        elif method == "seasonal":
            return self._predict_seasonal_naive(steps)
        else:
            raise ValueError(f"Unknown method '{method}'")

    def predict_detail(self, steps: int = 1, reference: type(self) | None = None, TA: float = 0) -> dict:
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
            result["exact_polynomial"] = poly_exact

        geo_exact = self._predict_geometric(steps)
        if geo_exact:
            result["exact_geometric"] = geo_exact

        ratio_exact = self._predict_ratio_pattern(steps)
        if ratio_exact:
            result["exact_ratio_pattern"] = ratio_exact

        # Add function mappings if reference provided
        if reference is not None and TA > 0:
            n = len(self._data)
            x_history = reference._data[:n]
            y_history = self._data
            deadline = time.time() + TA
            mappings = self._discover_mappings(x_history, y_history, deadline, precision=4)
            result["function_mappings"] = [
                {"function": desc, "fit_error": err} for _, err, desc in mappings[:10]
            ]

        return result