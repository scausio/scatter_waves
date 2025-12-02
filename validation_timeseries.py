"""
Wave Model Validation - Time Series Analysis Module

This module generates comprehensive time series plots for wave model validation,
including mean wave height evolution, error metrics over time, and percentile-based
extreme event analysis.

Features
--------
- Multi-model comparison time series
- Daily/aggregated statistics with confidence intervals
- Error metrics evolution (NRMSE, NBIAS, MAE)
- Percentile-based validation for extreme events
- Publication-quality graphics with enhanced styling
- Rolling statistics and trend analysis

Author: Salvatore Causio
Date: 2025
"""

import matplotlib

matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from stats import metrics
from utils import getConfigurationByID
import seaborn as sns
from matplotlib.dates import DateFormatter
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


def timeseries(ds: xr.Dataset, conf, outname: str,
               percentile_thresholds: Optional[List[int]] = None,
               **kwargs) -> None:
    """
    Generate comprehensive time series validation plots.

    Creates multiple visualization panels:
    1. Main time series comparing model(s) and observations
    2. Error metrics evolution (NRMSE, NBIAS, MAE)
    3. Optional percentile-based extreme event validation

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing model_hs, hs, and day variables
    conf : object
        Configuration object with experiment details and filters
    outname : str
        Base filename for output plots
    percentile_thresholds : list of int, optional
        Percentiles for extreme value validation (e.g., [90, 95])
    **kwargs : dict
        Additional parameters:
        - title: Plot title
        - rolling_window: Days for rolling mean (default: None)

    Notes
    -----
    Generates multiple output files:
    - Main time series: {outname}
    - Statistics time series: {outname.replace('Avg', '_stats_')}
    - Percentile analysis: {outname.replace('Avg', '_percentile_')} (if requested)
    """
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    datasets = conf.experiments

    # Calculate percentile thresholds if requested
    if percentile_thresholds:
        data_allmodels = ds.groupby('day').apply(
            lambda x: metrics(x, percentile_thresholds=percentile_thresholds))
    else:
        data_allmodels = ds.groupby('day').apply(metrics)

    print("Computed daily metrics:")
    print(data_allmodels)

    # Enhanced color palette
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
              '#1ABC9C', '#E67E22', '#34495E']

    # Get rolling window from kwargs
    rolling_window = kwargs.get('rolling_window', None)

    # ==========================================
    # PLOT 1: MAIN TIME SERIES WITH CONFIDENCE INTERVALS
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Plot model data with enhanced styling and confidence intervals
    for i, dataset in enumerate(datasets):
        data = data_allmodels.sel(model=dataset)

        # Main line
        line = ax.plot(data.day, data['model_hs'],
                       color=colors[i % len(colors)],
                       label=dataset.upper(),
                       linewidth=2.8,
                       alpha=0.85,
                       marker='o',
                       markersize=5,
                       markevery=max(1, len(data.day) // 25),
                       zorder=5)

        # Add confidence interval (± std)
        if 'model_std' in data:
            ax.fill_between(data.day,
                            data['model_hs'] - data['model_std'],
                            data['model_hs'] + data['model_std'],
                            alpha=0.15,
                            color=colors[i % len(colors)],
                            zorder=1)

        # Add rolling mean if requested
        if rolling_window:
            rolling_mean = data['model_hs'].rolling(day=rolling_window, center=True).mean()
            ax.plot(data.day, rolling_mean,
                    color=colors[i % len(colors)],
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.6,
                    label=f'{dataset} (rolling {rolling_window}d)')

    # Handle bias correction for satellite data
    if hasattr(conf.filters, 'unbias') and conf.filters.unbias in ['True', 'T', 'TRUE', 't']:
        data['sat_hs'] -= np.nanmean(data['sat_hs'])
        data['sat_hs'] += np.nanmean(data['model_hs'])
        sat_label = 'Satellite (Unbiased)'
    else:
        sat_label = 'Satellite'

    # Plot satellite data
    ax.plot(data.day, data['sat_hs'],
            color='#2C3E50',
            label=sat_label,
            linewidth=3.5,
            alpha=0.95,
            linestyle='-',
            marker='s',
            markersize=4,
            markevery=max(1, len(data.day) // 30),
            zorder=10)

    # Add satellite confidence interval
    if 'sat_std' in data:
        ax.fill_between(data.day,
                        data['sat_hs'] - data['sat_std'],
                        data['sat_hs'] + data['sat_std'],
                        alpha=0.15,
                        color='#2C3E50',
                        zorder=2)

    # Enhanced styling
    ax.set_ylabel('Significant Wave Height [m]', fontsize=15, fontweight='bold')
    ax.set_xlabel('Date', fontsize=15, fontweight='bold')

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize=17, fontweight='bold', pad=20)

    # Improve grid
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.9)
    ax.set_axisbelow(True)

    # Enhanced legend with statistics
    legend_elements = ax.get_legend_handles_labels()[0]
    legend_labels = ax.get_legend_handles_labels()[1]

    # Add mean values to legend
    for i, dataset in enumerate(datasets):
        data_model = data_allmodels.sel(model=dataset)
        mean_val = float(data_model['model_hs'].mean())
        legend_labels[i] = f'{legend_labels[i]} (μ={mean_val:.2f}m)'

    # Satellite mean
    sat_mean = float(data['sat_hs'].mean())
    legend_labels[-1] = f'{legend_labels[-1]} (μ={sat_mean:.2f}m)'

    legend = ax.legend(legend_elements, legend_labels,
                       loc='upper left', frameon=True, fancybox=True,
                       shadow=True, fontsize=11, framealpha=0.95,
                       ncol=1 if len(datasets) <= 4 else 2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#CCCCCC')

    # Format x-axis dates
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Main time series saved: {outname}")

    # ==========================================
    # PLOT 2: ERROR METRICS EVOLUTION
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('white')

    if 'title' in kwargs:
        fig.suptitle(f"{kwargs['title']} - Error Metrics Evolution",
                     fontsize=17, fontweight='bold', y=0.995)

    # Set background colors
    for ax in axes:
        ax.set_facecolor('#FAFAFA')

    metric_names = ['nrmse', 'nbias', 'mae']
    metric_labels = ['NRMSE [%]', 'NBIAS [%]', 'MAE [m]']
    metric_colors = ['#E74C3C', '#3498DB', '#2ECC71']

    # Plot each metric
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]

        for i, dataset in enumerate(datasets):
            data = data_allmodels.sel(model=dataset)

            # Convert to percentage for normalized metrics
            if metric in ['nrmse', 'nbias']:
                values = data[metric] * 100
            else:
                values = data[metric]

            # Calculate mean for legend
            mean_val = float(np.nanmean(values))

            # Main line
            ax.plot(data.day, values,
                    color=colors[i % len(colors)],
                    label=f"{dataset.upper()}: μ={mean_val:.3f}",
                    linewidth=2.5,
                    alpha=0.85,
                    marker='o',
                    markersize=4,
                    markevery=max(1, len(data.day) // 25))

            # Add rolling mean if requested
            if rolling_window:
                rolling_mean = values.rolling(day=rolling_window, center=True).mean()
                ax.plot(data.day, rolling_mean,
                        color=colors[i % len(colors)],
                        linestyle='--',
                        linewidth=1.8,
                        alpha=0.5)

        # Styling
        ax.set_ylabel(label, fontsize=14, fontweight='bold')

        # Add reference line at zero for bias
        if metric == 'nbias':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

        # Grid
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.9)
        ax.set_axisbelow(True)

        # Legend
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                           shadow=True, fontsize=10, framealpha=0.95,
                           ncol=1 if len(datasets) <= 3 else 2)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#CCCCCC')

        # Tick parameters
        ax.tick_params(axis='y', labelsize=11)

        # Borders
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.5)

    # X-axis label only on bottom plot
    axes[-1].set_xlabel('Date', fontsize=14, fontweight='bold')
    axes[-1].tick_params(axis='x', rotation=45, labelsize=11)

    plt.tight_layout()
    stats_outname = outname.replace('Avg', '_stats_')
    plt.savefig(stats_outname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Error metrics time series saved: {stats_outname}")

    # ==========================================
    # PLOT 3: PERCENTILE-BASED VALIDATION (if requested)
    # ==========================================
    if percentile_thresholds:
        fig, axes = plt.subplots(len(percentile_thresholds), 1,
                                 figsize=(14, 5 * len(percentile_thresholds)),
                                 sharex=True)
        fig.patch.set_facecolor('white')

        if 'title' in kwargs:
            fig.suptitle(f"{kwargs['title']} - Extreme Event Validation (Percentiles)",
                         fontsize=17, fontweight='bold', y=0.995)

        # Ensure axes is iterable
        if len(percentile_thresholds) == 1:
            axes = [axes]

        for idx, percentile in enumerate(percentile_thresholds):
            ax = axes[idx]
            ax.set_facecolor('#FAFAFA')

            for i, dataset in enumerate(datasets):
                data = data_allmodels.sel(model=dataset)

                # Check if percentile metrics exist
                if f'bias_p{percentile}' in data:
                    bias_p = data[f'bias_p{percentile}']
                    rmse_p = data[f'rmse_p{percentile}']
                    nobs_p = data[f'nobs_p{percentile}']

                    # Plot BIAS for this percentile
                    ax.plot(data.day, bias_p,
                            color=colors[i % len(colors)],
                            label=f"{dataset.upper()} BIAS",
                            linewidth=2.5,
                            alpha=0.85,
                            marker='o',
                            markersize=4,
                            markevery=max(1, len(data.day) // 25))

                    # Plot RMSE for this percentile
                    ax.plot(data.day, rmse_p,
                            color=colors[i % len(colors)],
                            label=f"{dataset.upper()} RMSE",
                            linewidth=2.5,
                            alpha=0.85,
                            linestyle='--',
                            marker='^',
                            markersize=4,
                            markevery=max(1, len(data.day) // 25))

            # Styling
            ax.set_ylabel(f'P{percentile} Error [m]', fontsize=14, fontweight='bold')
            ax.set_title(f'Validation for ≥{percentile}th Percentile Events',
                         fontsize=13, fontweight='bold', pad=10)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.9)
            ax.set_axisbelow(True)

            # Legend
            legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                               shadow=True, fontsize=10, framealpha=0.95,
                               ncol=len(datasets))
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('#CCCCCC')

            # Borders
            for spine in ax.spines.values():
                spine.set_edgecolor('#CCCCCC')
                spine.set_linewidth(1.5)

            ax.tick_params(axis='y', labelsize=11)

        # X-axis label
        axes[-1].set_xlabel('Date', fontsize=14, fontweight='bold')
        axes[-1].tick_params(axis='x', rotation=45, labelsize=11)

        plt.tight_layout()
        percentile_outname = outname.replace('Avg', '_percentile_')
        plt.savefig(percentile_outname, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Percentile validation time series saved: {percentile_outname}")


def main(conf_path: str, start_date: str, end_date: str) -> None:
    """
    Main function for time series validation analysis.

    Generates comprehensive time series plots comparing model predictions
    with satellite observations, including error metrics evolution and
    optional percentile-based extreme event validation.

    Parameters
    ----------
    conf_path : str
        Path to configuration YAML file
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format

    Notes
    -----
    Configuration file should contain:
    - experiments: Dictionary of model experiments
    - out_dir: Output directory path
    - filters: Quality control parameters
    - title: Plot title template
    - percentile_thresholds (optional): List of percentiles for extreme validation

    Outputs
    -------
    Generates three types of plots in the output directory:
    1. timeseriesAvg_*.jpeg - Main time series with wave heights
    2. timeseries_stats_*.jpeg - Error metrics evolution
    3. timeseries_percentile_*.jpeg - Percentile-based validation (if configured)

    Examples
    --------
    >>> main('conf.yaml', '20230901', '20230930')
    """
    # Load configuration
    conf = getConfigurationByID(conf_path, 'plot')
    outdir = os.path.join(conf.out_dir.out_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    date = f"{start_date}_{end_date}"

    print("=" * 80)
    print(f"TIME SERIES VALIDATION: {start_date} to {end_date}")
    print("=" * 80)

    # Get percentile thresholds from configuration
    percentile_thresholds = getattr(conf.filters, 'percentile_thresholds', None)
    if percentile_thresholds:
        print(f"Percentile thresholds configured: {percentile_thresholds}")

    # Load dataset (use first experiment to get data structure)
    for dataset in conf.experiments:
        ds_path = (conf.experiments[dataset].series).format(
            out_dir=conf.out_dir.out_dir, date=date, experiment=dataset)
        ds_all = xr.open_dataset(ds_path)

        # Add day coordinate for grouping
        ds_all['day'] = ('obs', ds_all.time.dt.floor('d').values)

        print(f"\nDataset loaded: {dataset}")
        print(f"  Time range: {ds_all.time.min().values} to {ds_all.time.max().values}")
        print(f"  Total observations: {len(ds_all.obs)}")
        print(f"  Unique days: {len(np.unique(ds_all['day']))}")

        # Only need to load once
        break

    # Generate output filename
    outName = os.path.join(outdir, f'timeseriesAvg_{dataset}_{date}.jpeg')

    # Generate time series plots
    print(f"\nGenerating time series plots...")
    timeseries(ds_all, conf, outName,
               percentile_thresholds=percentile_thresholds,
               title=f"{conf.title}".format(start_date=start_date, end_date=end_date),
               rolling_window=None)  # Set to integer (e.g., 7) for rolling mean

    print(f"\n{'=' * 80}")
    print("TIME SERIES VALIDATION COMPLETE")
    print(f"Output directory: {outdir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Wave Model Validation - Time Series Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--config', required=True,
                        help='Path to configuration YAML file')

    args = parser.parse_args()

    print("Note: This script is typically called from main.py")
