"""Statistics shared by the analyses: rank correlations, intervals, binomial helpers, veto."""

import math

import numpy as np
from scipy import stats as scipy_stats

INTERVAL_LEVEL = 0.95  # confidence level of every Fisher-z interval


def spearman(x, y) -> float:
    """Return Spearman's rho of two sequences (average ranks for ties)."""
    return float(scipy_stats.spearmanr(x, y).statistic)


def partial_spearman(x, y, z) -> float:
    """Return the partial Spearman rho of x and y given z, from the three pairwise coefficients.

    Defined only when neither x nor y is perfectly rank-correlated with z (|rho| < 1).
    """
    rho_xy = spearman(x, y)
    rho_xz = spearman(x, z)
    rho_yz = spearman(y, z)
    return (rho_xy - rho_xz * rho_yz) / math.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))


def fisher_interval(rho: float, n: int, controls: int = 0) -> tuple:
    """Return the 95% Fisher-z interval of a correlation over n points, given `controls` controls.

    The interval is tanh(atanh(rho) +- z / sqrt(n - 3 - controls)). It is defined for |rho| < 1
    and n > 3 + controls.
    """
    z_value = scipy_stats.norm.ppf(0.5 + INTERVAL_LEVEL / 2)
    half_width = z_value / math.sqrt(n - 3 - controls)
    center = math.atanh(rho)
    return math.tanh(center - half_width), math.tanh(center + half_width)


def binomial_se(rate: float, episodes: int) -> float:
    """Return the standard error of a success rate measured on `episodes` episodes."""
    return math.sqrt(rate * (1 - rate) / episodes)


def expected_max_binomial(rates, episodes: int) -> float:
    """Return E[max_k X_k / episodes] for independent X_k ~ Binomial(episodes, rate_k), exactly."""
    counts = np.arange(episodes + 1)
    cdf_of_max = np.ones(episodes + 1)
    for rate in rates:
        cdf_of_max = cdf_of_max * scipy_stats.binom.cdf(counts, episodes, rate)
    return float(np.sum(1 - cdf_of_max[:-1])) / episodes


def shrink_near_ties(rates, band: float, episodes: int) -> np.ndarray:
    """Shrink the rates within `band` of the top rate toward their mean (method of moments).

    The between-checkpoint variance of the near-tied rates is their sample variance minus the
    mean binomial variance; each near-tied rate keeps that share of its distance from the mean.
    Rates farther than `band` below the top are returned unchanged.
    """
    rates = np.asarray(rates, dtype=float)
    near = np.flatnonzero(rates.max() - rates <= band)
    if len(near) < 2:
        return rates.copy()
    noise_variance = float(np.mean(rates[near] * (1 - rates[near]) / episodes))
    true_variance = max(0.0, float(np.var(rates[near], ddof=1)) - noise_variance)
    total = true_variance + noise_variance
    keep = true_variance / total if total > 0 else 0.0
    shrunk = rates.copy()
    near_mean = rates[near].mean()
    shrunk[near] = near_mean + keep * (rates[near] - near_mean)
    return shrunk


def median_by_step(series: list[dict]) -> dict[int, float]:
    """Return the median over series at every step that all of them have, in step order."""
    common = sorted(set.intersection(*(set(values) for values in series)))
    if not common:
        raise ValueError("the series share no step, so there is no median at any step")
    return {step: float(np.median([values[step] for values in series])) for step in common}


def spread_veto_pick(spread_by_step: dict, mads: float) -> int:
    """Return the latest checkpoint whose ensemble spread does not exceed the veto threshold.

    The threshold is the median spread plus `mads` raw median absolute deviations (no 1.4826
    factor). At least half the checkpoints lie at or below the median, so some checkpoint always
    survives.
    """
    spreads = np.asarray(list(spread_by_step.values()), dtype=float)
    median = np.median(spreads)
    threshold = float(median + mads * np.median(np.abs(spreads - median)))
    return max(step for step, spread in spread_by_step.items() if spread <= threshold)
