"""Test harness for Dive module."""

import math

from main import Dive


def _benchmark_predict_next(num_series: int = 1000, steps: int = 1) -> None:
    """Evaluate predict_next accuracy across many synthetic series."""
    import random

    methods = [
        "ensemble",
        "linear",
        "quadratic",
        "holt",
        "exponential",
        "differences",
        "drift",
        "newton",
        "lagrange",
        "seasonal",
    ]

    def synthetic_functions():
        yield ("linear", lambda x: 1.2 * x + 0.5)
        yield ("quadratic", lambda x: 0.03 * x**2 - 0.4 * x + 2.0)
        yield ("cubic", lambda x: 0.0015 * x**3 - 0.02 * x**2 + 0.5 * x + 1.0)
        yield ("exponential", lambda x: 1.1 ** x + 0.1 * x)
        yield ("geometric", lambda x: 2.5 * (1.15 ** x))
        yield ("sinusoidal", lambda x: 5.0 * math.sin(0.25 * x + 0.3) + 10.0)
        yield ("periodic", lambda x: 3.0 * (x % 5) + 10.0)
        yield ("mixed", lambda x: 0.02 * x**2 + 0.5 * x + 3.0 + ((-1) ** int(round(x))) * 0.5)

    stats = {m: {"count": 0, "mae": 0.0, "mape": 0.0, "mse": 0.0} for m in methods}

    random.seed(42)
    for func_name, f in synthetic_functions():
        for _ in range(num_series // 8):
            start = random.uniform(-5, 5)
            scale = random.uniform(0.8, 1.25)
            n = random.randint(12, 32)
            data = [f(scale * (i + start)) for i in range(n)]

            noised = [x * (1 + random.uniform(-0.02, 0.02)) for x in data]
            d = Dive(noised)
            actual = f(scale * (n + start))

            for method in methods:
                try:
                    pred = d.predict_next(steps=steps, method=method)
                    pred_val = pred[0] if isinstance(pred, list) else pred
                except Exception:
                    continue

                err = abs(pred_val - actual)
                relat = abs(err / actual) if abs(actual) > 1e-9 else err

                stats[method]["count"] += 1
                stats[method]["mae"] += err
                stats[method]["mse"] += err**2
                stats[method]["mape"] += relat

    print("\n=== predict_next benchmark results ===")
    for method in methods:
        cnt = stats[method]["count"]
        if cnt == 0:
            print(f"{method}: no data")
            continue
        mae_avg = stats[method]["mae"] / cnt
        rmse = math.sqrt(stats[method]["mse"] / cnt)
        mape_avg = 100.0 * stats[method]["mape"] / cnt
        print(f"{method:10s}: N={cnt}  MAE={mae_avg:.6f}  RMSE={rmse:.6f}  MAPE={mape_avg:.2f}%")


def _sanity_check() -> None:
    d = Dive([3, 1, 4, 1, 5, 9, 2, 6])
    print("mean", d.mean())
    print("median", d.median())
    print("sum", d.sum())
    d += 7
    print("predict_next (default ensemble)", d.predict_next())
    print(d.summary())


if __name__ == "__main__":
    _sanity_check()
    _benchmark_predict_next(num_series=1600, steps=1)
