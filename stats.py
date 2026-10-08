"""
Statistical metrics for wave model validation.

Scalar metrics take model (data) and observation arrays; NaNs are ignored.
`grouped_metrics` computes the same metrics for many groups at once (grid
cells, days, satellites) with np.bincount, without Python loops over groups.
"""
import numpy as np


def _valid(data, obs):
    data = np.asarray(data, dtype='f8')
    obs = np.asarray(obs, dtype='f8')
    ok = np.isfinite(data) & np.isfinite(obs)
    return data[ok], obs[ok]


def BIAS(data, obs):
    """mean(model - obs): > 0 means overestimation."""
    data, obs = _valid(data, obs)
    return np.round(np.mean(data - obs), 4) if len(obs) else np.nan


def RMSE(data, obs):
    data, obs = _valid(data, obs)
    return np.round(np.sqrt(np.mean((data - obs) ** 2)), 3) if len(obs) else np.nan


def MAE(data, obs):
    data, obs = _valid(data, obs)
    return np.round(np.mean(np.abs(data - obs)), 3) if len(obs) else np.nan


def ScatterIndex(data, obs):
    """sqrt(sum(((m - <m>) - (o - <o>))^2) / sum(o^2)): unbiased scatter index (Mentaschi et al., 2013)."""
    data, obs = _valid(data, obs)
    if len(obs) == 0:
        return np.nan
    num = np.sum(((data - data.mean()) - (obs - obs.mean())) ** 2)
    return np.round(np.sqrt(num / np.sum(obs ** 2)), 3)


def correlation(data, obs):
    data, obs = _valid(data, obs)
    if len(obs) < 2:
        return np.nan
    return np.round(np.corrcoef(data, obs)[0, 1], 4)


def skill_score(data, obs, reference=None):
    """Murphy skill score 1 - MSE / MSE_reference (reference: observed mean)."""
    data, obs = _valid(data, obs)
    if len(obs) == 0:
        return np.nan
    if reference is None:
        reference = obs.mean()
    mse_reference = np.mean((reference - obs) ** 2)
    if mse_reference == 0:
        return np.nan
    return np.round(1 - np.mean((data - obs) ** 2) / mse_reference, 4)


def symmetric_slope(data, obs):
    """Mean of the OLS slope of model on obs and the inverse slope of obs on model."""
    data, obs = _valid(data, obs)
    if len(obs) < 2:
        return np.nan
    cov = np.sum((obs - obs.mean()) * (data - data.mean()))
    forward = cov / np.sum((obs - obs.mean()) ** 2)
    backward_inv = cov / np.sum((data - data.mean()) ** 2)
    backward = 1 / backward_inv if backward_inv != 0 else np.nan
    return np.round((forward + backward) / 2, 4)


def quantile_bias(data, obs, quantile=0.9):
    """Difference between model and observed quantile (distribution, not paired)."""
    data, obs = _valid(data, obs)
    if len(obs) < 2:
        return np.nan
    return np.round(np.quantile(data, quantile) - np.quantile(obs, quantile), 4)


def summary(data, obs):
    """All scalar metrics in a dict (normalised metrics as fractions)."""
    data, obs = _valid(data, obs)
    n = len(obs)
    if n == 0:
        return {'n': 0}
    mean_obs = obs.mean()
    corr = correlation(data, obs)
    return {
        'n': n,
        'bias': np.mean(data - obs),
        'rmse': np.sqrt(np.mean((data - obs) ** 2)),
        'mae': np.mean(np.abs(data - obs)),
        'nbias': np.mean(data - obs) / mean_obs,
        'nrmse': np.sqrt(np.sum((data - obs) ** 2) / np.sum(obs ** 2)),
        'si': ScatterIndex(data, obs),
        'correlation': corr,
        'r2': corr ** 2,
        'skill_score': skill_score(data, obs),
        'symmetric_slope': symmetric_slope(data, obs),
        'mean_obs': mean_obs,
        'mean_model': data.mean(),
        'std_obs': obs.std(),
        'std_model': data.std(),
        'p95_obs': np.percentile(obs, 95),
        'p95_model': np.percentile(data, 95),
    }


def grouped_metrics(data, obs, groups, n_groups, min_obs=1):
    """
    Metrics for each group label in [0, n_groups) (e.g. flattened grid cell
    index, day index). Groups with fewer than min_obs pairs are NaN.

    Returns a dict of arrays of length n_groups.
    """
    data = np.asarray(data, dtype='f8')
    obs = np.asarray(obs, dtype='f8')
    ok = np.isfinite(data) & np.isfinite(obs) & (groups >= 0)
    g, m, o = groups[ok], data[ok], obs[ok]
    d = m - o

    def s(w=None):
        return np.bincount(g, weights=w, minlength=n_groups).astype('f8')

    n = s()
    with np.errstate(invalid='ignore', divide='ignore'):
        sm, so, sd = s(m), s(o), s(d)
        smm, soo, smo, sdd = s(m * m), s(o * o), s(m * o), s(d * d)
        mean_m, mean_o = sm / n, so / n
        var_m = np.maximum(smm / n - mean_m ** 2, 0)
        var_o = np.maximum(soo / n - mean_o ** 2, 0)
        cov = smo / n - mean_m * mean_o
        out = {
            'nobs': n,
            'bias': sd / n,
            'rmse': np.sqrt(sdd / n),
            'mae': s(np.abs(d)) / n,
            'nbias': sd / so,
            'nrmse': np.sqrt(sdd / soo),
            'si': np.sqrt(np.maximum(sdd / n - (sd / n) ** 2, 0) / (soo / n)),
            'correlation': np.where(n > 2, cov / np.sqrt(var_m * var_o), np.nan),
            'model_hs': mean_m,
            'sat_hs': mean_o,
            'model_std': np.sqrt(var_m),
            'sat_std': np.sqrt(var_o),
        }
    too_few = n < max(min_obs, 1)
    for key in out:
        if key != 'nobs':
            out[key][too_few] = np.nan
    return out
