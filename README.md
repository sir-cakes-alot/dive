# DIVE — Data Insight and Visualization Engine

`dive-for-data` provides `Dive`, a small pure-Python container for ordered numeric data with:
- **Statistical summaries** (`mean`, `median`, `mode`, `stdev`, `variance`, `skewness`, `kurtosis`, etc.)
- **Quantiles and histograms** (`percentile`, `quartiles`, `iqr`)
- **Data transforms** (`z_scores`, `normalized`, `cumulative_sum`, `moving_average`, `diff`, `pct_change`, `sorted`, `clip`, `apply`)
- **Prediction engine** (`predict_next`, `predict_detail`, `linear`, `quadratic`, `holt`, `exponential`, `drift`, `newton`, `lagrange`, `seasonal`, `ensemble`)
- **Regression/Correlation analysis** (`correlation`, `covariance`, `regress_on`)
- **ASCII visualizations** (`histogram`, `sparkline`, `plot_ascii`)
- **Utility exports** (`to_list`, `to_dict`)

---

## Installation

```bash
pip install dive-for-data
```

## Quickstart

```python
from dive import Dive

# Create dataset
sales = Dive([100, 150, 120, 200, 180])
print(sales.mean())          # 150.0
print(sales.summary())

# Add new value
sales += 220
print(sales[-1])            # 220.0

# Predict next value (ensemble model)
print(sales.predict_next())

# Use reference series for regression mode (len(reference) == len(self) + steps)
# This learns the relationship F(temps) -> sales
temps = Dive([20, 25, 22, 30, 28, 35])
print(sales.predict_next(reference=temps, TA=1))

# Detailed prediction report
report = sales.predict_detail(steps=3, reference=temps, TA=1)
print(report["ensemble"])
``` 

## API overview

### Data management
- `Dive(data=None)`
- `add`, `append`, `remove`, `pop`, `clear`, `copy`, `data` property
- supports Python protocols: `len`, indexing, iteration, `in`, equality

### Stats
- `mean`, `median`, `mode`, `geo_mean`, `harmonic_mean`
- `stdev`, `variance`, `range`, `min`, `max`, `sum`
- `percentile`, `quartiles`, `iqr`.

### Prediction
- `predict_next(steps=1, method='ensemble', reference=None, corr_threshold=0.1, TA=0)`
- `predict_detail(...)`

### Cross-dataset
- `correlation(other)`
- `covariance(other)`
- `regress_on(other)`

### Visualization
- `summary()` / `describe()`
- `histogram(bins=10, width=40)`
- `sparkline()`
- `plot_ascii(width=60, height=15)`

## License

This project is licensed under the GNU General Public License v3 (GPLv3).
