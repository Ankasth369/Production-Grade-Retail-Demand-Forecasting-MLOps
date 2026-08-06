import numpy as np
from scipy import stats


def psi(reference, current, bins=10):
    """Population Stability Index between two distributions."""
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference, current):
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)
