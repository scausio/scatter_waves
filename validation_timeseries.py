"""
Time series validation: metrics aggregated per period (default daily).

The domain-mean Hs of a day depends on which tracks crossed the domain, so the
number of pairs per period is shown together with the errors.

Configuration (plot section):
  timeseries_freq : pandas frequency for the aggregation ('D', 'W', 'MS', ...; default 'D')
  filters.percentile_thresholds : bias/rmse time series for obs >= global percentile
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pairs import load_pairs, load_plot_conf, plot_dir
from stats import grouped_metrics

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']


def period_metrics(pairs, freq='D', min_obs=1, threshold=None):
    """{model: DataFrame indexed by period} with the grouped metrics."""
    periods = pd.DatetimeIndex(pairs.time.values).to_period(freq)
    labels, uniques = pd.factorize(periods, sort=True)
    obs = pairs.hs.values
    sel = obs >= threshold if threshold is not None else np.ones(len(obs), bool)
    out = {}
    for model in pairs.model.values:
        mod = pairs.model_hs.sel(model=model).values
        res = grouped_metrics(mod[sel], obs[sel], labels[sel], len(uniques), min_obs)
        out[model] = pd.DataFrame(res, index=uniques.to_timestamp())
    return out


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)


def plot_timeseries(metrics, outname, title='', freq='D'):
    models = list(metrics)
    first = metrics[models[0]]
    marker = dict(marker='o', markersize=3)

    # mean Hs of the collocated pairs
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    ax.plot(first.index, first['sat_hs'], color='black', lw=2.5, label='Satellite', **marker)
    for i, model in enumerate(models):
        ax.plot(first.index, metrics[model]['model_hs'], color=COLORS[i % len(COLORS)], lw=2,
                label=model, **marker)
    _style(ax, 'Mean Hs of pairs [m]')
    ax.legend(loc='upper left')
    ax.set_title(title, fontsize=15, fontweight='bold')
    axn.bar(first.index, first['nobs'], width=0.8 if freq == 'D' else 5, color='gray')
    _style(axn, 'N pairs')
    fig.autofmt_xdate()
    fig.savefig(outname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # error metrics
    panels = [('bias', 'Bias [m]', 1), ('rmse', 'RMSE [m]', 1),
              ('nbias', 'NBIAS [%]', 100), ('nrmse', 'NRMSE [%]', 100)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 13), sharex=True)
    for ax, (key, label, scale) in zip(axes, panels):
        for i, model in enumerate(models):
            v = metrics[model][key] * scale
            ax.plot(v.index, v, color=COLORS[i % len(COLORS)], lw=2,
                    label=f'{model}: mean {np.nanmean(v):.3f}', **marker)
        if key in ('bias', 'nbias'):
            ax.axhline(0, color='gray', lw=1)
        _style(ax, label)
        ax.legend(loc='upper left', fontsize=9)
    axes[0].set_title(f'{title} - error metrics', fontsize=15, fontweight='bold')
    fig.autofmt_xdate()
    fig.savefig(outname.replace('timeseriesAvg', 'timeseries_stats'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_percentile_timeseries(pairs, percentiles, outname, title='', freq='D'):
    fig, axes = plt.subplots(len(percentiles), 2, figsize=(16, 4 * len(percentiles)),
                             sharex=True, squeeze=False)
    for row, p in enumerate(percentiles):
        thr = np.percentile(pairs.hs.values, p)
        metrics = period_metrics(pairs, freq, threshold=thr)
        for i, (model, df) in enumerate(metrics.items()):
            for col, key in enumerate(('bias', 'rmse')):
                axes[row, col].plot(df.index, df[key], 'o-', ms=3, color=COLORS[i % len(COLORS)],
                                    label=f'{model}: mean {np.nanmean(df[key]):.3f}')
        for col, key in enumerate(('Bias', 'RMSE')):
            ax = axes[row, col]
            _style(ax, f'{key} [m]')
            ax.set_title(f'obs ≥ P{p} ({thr:.2f} m)', fontsize=11)
            ax.legend(fontsize=9)
            if key == 'Bias':
                ax.axhline(0, color='gray', lw=1)
    fig.suptitle(f'{title} - extreme events (global percentiles)', fontsize=15, fontweight='bold')
    fig.autofmt_xdate()
    fig.savefig(outname.replace('timeseriesAvg', 'timeseries_percentile'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main(conf_path, start_date, end_date):
    conf = load_plot_conf(conf_path)
    outdir = plot_dir(conf)
    date = f'{start_date}_{end_date}'
    freq = conf.get('timeseries_freq', 'D')
    title = conf.title.format(start_date=start_date, end_date=end_date)

    pairs = load_pairs(conf, start_date, end_date, verbose=False)
    metrics = period_metrics(pairs, freq)
    outname = os.path.join(outdir, f'timeseriesAvg_{date}.jpeg')
    plot_timeseries(metrics, outname, title, freq)
    percentiles = conf.filters.get('percentile_thresholds') or []
    if percentiles:
        plot_percentile_timeseries(pairs, percentiles, outname, title, freq)
    pd.concat(metrics, names=['model', 'period']).to_csv(
        os.path.join(outdir, f'timeseries_metrics_{date}.csv'), float_format='%.4f')
    print(f'time series saved in {outdir}')
