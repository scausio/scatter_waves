"""
Wave Model Validation - Satellite Track Analysis Module

This module generates spatial plots showing satellite tracks colored by
validation metrics. Useful for visualizing along-track model performance
and identifying track-specific patterns or issues.

Features
--------
- Satellite track visualization with metric overlay
- Daily track analysis
- Multiple metrics support (BIAS, RMSE, NRMSE, NBIAS, MAE)
- High-resolution coastline overlay
- Publication-quality geographic plots

Author: Salvatore Causio
Date: 2025
"""

import matplotlib

matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import getConfigurationByID, ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings('ignore')
from typing import Optional


def varConversion(variable: xr.DataArray, variable_name: str) -> np.ndarray:
    """
    Convert variable to appropriate units for visualization.

    Normalized metrics (NRMSE, NBIAS) are converted to percentages.

    Parameters
    ----------
    variable : xr.DataArray
        Variable to convert
    variable_name : str
        Name of the variable (determines conversion)

    Returns
    -------
    np.ndarray
        Converted variable values
    """
    if variable_name in ['bias', 'rmse', 'mae']:
        return variable.values
    elif variable_name in ['nbias', 'nrmse', 'nmae']:
        return variable.values * 100
    else:
        return variable.values


def plotTracks(ds: xr.Dataset, variable: str, coast_resolution: str,
               outfile: str, daily_track: bool = False) -> None:
    """
    Plot satellite tracks colored by validation metric.

    Creates geographic visualization showing satellite ground tracks
    with points colored according to the specified validation metric.
    Useful for identifying spatial patterns in model performance and
    track-specific issues.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing latitude, longitude, time, and validation metrics
    variable : str
        Validation metric to plot ('bias', 'rmse', 'nrmse', 'nbias', 'mae', etc.)
    coast_resolution : str
        Basemap coastline resolution ('c', 'l', 'i', 'h', 'f')
        c=crude, l=low, i=intermediate, h=high, f=full
    outfile : str
        Path for output file
    daily_track : bool, optional
        If True, generate separate plot for each day (default: False)

    Notes
    -----
    Color scales are optimized for each metric type:
    - Bias metrics: diverging colormap (blue-white-red)
    - Error metrics: sequential colormap (yellow-orange-red)

    For daily tracks, individual files are generated with date suffix.
    """
    # Extract coordinates
    x = ds.longitude.values
    y = ds.latitude.values

    # Create figure
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.gca()

    # Create basemap
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())

    m = Basemap(llcrnrlon=lon_min - 0.25, llcrnrlat=lat_min - 0.25,
                urcrnrlat=lat_max + 0.25, urcrnrlon=lon_max + 0.25,
                resolution=coast_resolution,
                projection='merc')

    # Draw map features
    m.drawcoastlines(linewidth=1.2, color='#333333')
    m.fillcontinents(color='#E0E0E0', lake_color='#B0D4E3')
    m.drawmapboundary(fill_color='#B0D4E3')

    # Draw grid lines
    tick_spacing = ticker(lat_min, lat_max)
    meridians = np.arange(-180, 180 + tick_spacing, tick_spacing)
    parallels = np.arange(-90, 90 + tick_spacing, tick_spacing)
    m.drawparallels(parallels, labels=[True, False, False, True],
                    linewidth=0.5, fontsize=10, color='gray')
    m.drawmeridians(meridians, labels=[True, False, False, True],
                    linewidth=0.5, fontsize=10, color='gray')

    # Add day coordinate for daily tracking
    ds['day'] = ('obs', ds.time.dt.floor('d').values)

    # Configure variable-specific settings
    var_config = {
        'bias': {
            'title': 'Bias [m]',
            'vmin': -1.0,
            'vmax': 1.0,
            'cmap': 'RdBu_r',
            'marker_size': 30
        },
        'rmse': {
            'title': 'RMSE [m]',
            'vmin': 0,
            'vmax': 1.0,
            'cmap': 'YlOrRd',
            'marker_size': 30
        },
        'mae': {
            'title': 'MAE [m]',
            'vmin': 0,
            'vmax': 0.8,
            'cmap': 'YlOrRd',
            'marker_size': 30
        },
        'nrmse': {
            'title': 'NRMSE [%]',
            'vmin': 0,
            'vmax': 60,
            'cmap': 'viridis',
            'marker_size': 30
        },
        'nbias': {
            'title': 'NBIAS [%]',
            'vmin': -60,
            'vmax': 60,
            'cmap': 'RdBu_r',
            'marker_size': 30
        },
        'nmae': {
            'title': 'NMAE [%]',
            'vmin': 0,
            'vmax': 50,
            'cmap': 'YlOrRd',
            'marker_size': 30
        }
    }

    # Get configuration or use defaults
    if variable in var_config:
        config = var_config[variable]
    else:
        # Default configuration
        var_values = varConversion(ds[variable], variable)
        config = {
            'title': variable.upper(),
            'vmin': np.nanpercentile(var_values, 5),
            'vmax': np.nanpercentile(var_values, 95),
            'cmap': 'viridis',
            'marker_size': 30
        }

    # Calculate mean for title
    var_values = varConversion(ds[variable], variable)
    mean_val = np.nanmean(var_values)

    # Update title with mean
    title = f"{config['title']}: μ = {mean_val:.3f}"
    plt.title(title, loc='left', fontsize=14, fontweight='bold', pad=15)

    if daily_track:
        # Generate separate plot for each day
        unique_days = np.unique(ds['day'])
        print(f"  Generating {len(unique_days)} daily track plots...")

        for i, day in enumerate(unique_days):
            if i > 0:
                # Clear previous plot but keep map setup
                ax.clear()

                # Recreate basemap for new subplot
                m = Basemap(llcrnrlon=lon_min - 0.25, llcrnrlat=lat_min - 0.25,
                            urcrnrlat=lat_max + 0.25, urcrnrlon=lon_max + 0.25,
                            resolution=coast_resolution,
                            projection='merc')
                m.drawcoastlines(linewidth=1.2, color='#333333')
                m.fillcontinents(color='#E0E0E0', lake_color='#B0D4E3')
                m.drawmapboundary(fill_color='#B0D4E3')
                m.drawparallels(parallels, labels=[True, False, False, True],
                                linewidth=0.5, fontsize=10, color='gray')
                m.drawmeridians(meridians, labels=[True, False, False, True],
                                linewidth=0.5, fontsize=10, color='gray')

            # Select data for this day
            ds_daily = ds.isel(obs=ds['day'] == day)

            if len(ds_daily.obs) == 0:
                continue

            var_daily = varConversion(ds_daily[variable], variable)

            # Convert coordinates to map projection
            x_proj, y_proj = m(ds_daily.longitude.values, ds_daily.latitude.values)

            # Plot scatter with metric coloring
            im = ax.scatter(x_proj, y_proj, c=var_daily,
                            cmap=config['cmap'],
                            vmin=config['vmin'],
                            vmax=config['vmax'],
                            s=config['marker_size'],
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.5,
                            zorder=5)

            # Add colorbar (only on first iteration)
            if i == 0:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.15)
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(config['title'], fontsize=12,
                               fontweight='bold', labelpad=15)
                cbar.ax.tick_params(labelsize=10)

            # Update title with date
            day_str = str(day)[:10]
            mean_daily = np.nanmean(var_daily)
            title = f"{config['title']}: {day_str} (μ = {mean_daily:.3f})"
            plt.title(title, loc='left', fontsize=14, fontweight='bold', pad=15)

            # Add statistics textbox
            stats_text = f"N = {len(var_daily[~np.isnan(var_daily)])}\n"
            stats_text += f"μ = {mean_daily:.3f}\n"
            stats_text += f"σ = {np.nanstd(var_daily):.3f}"

            props = dict(boxstyle='round', facecolor='white',
                         alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', bbox=props,
                    family='monospace')

            # Save daily plot
            outfile_daily = outfile.replace('.jpeg', f'_{day_str}.jpeg')
            plt.savefig(outfile_daily, dpi=300, bbox_inches='tight',
                        facecolor='white')

            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(unique_days)} daily plots generated")

    else:
        # Plot all data at once
        print(f"  Generating combined track plot...")

        # Convert coordinates to map projection
        x_proj, y_proj = m(x, y)

        # Plot scatter with metric coloring
        im = ax.scatter(x_proj, y_proj, c=var_values,
                        cmap=config['cmap'],
                        vmin=config['vmin'],
                        vmax=config['vmax'],
                        s=config['marker_size'],
                        alpha=0.7,
                        edgecolors='white',
                        linewidth=0.5,
                        zorder=5)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(config['title'], fontsize=12,
                       fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=10)

        # Add statistics textbox
        valid_data = var_values[~np.isnan(var_values)]
        stats_text = f"N = {len(valid_data)}\n"
        stats_text += f"μ = {np.mean(valid_data):.3f}\n"
        stats_text += f"σ = {np.std(valid_data):.3f}\n"
        stats_text += f"min = {np.min(valid_data):.3f}\n"
        stats_text += f"max = {np.max(valid_data):.3f}"

        props = dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props,
                family='monospace')

        # Save plot
        plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')

    plt.close()
    print(f"  ✓ Track plot(s) saved")


def main(conf_path: str, start_date: str, end_date: str) -> None:
    """
    Main function for satellite track validation visualization.

    Generates geographic plots showing satellite ground tracks colored
    by validation metrics. Useful for identifying spatial patterns and
    along-track performance characteristics.

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
    - out_dir: Output directory
    - coast_resolution: Coastline detail level
    - filters: Quality control parameters (min, max, unbias)
    - daily_tracks (optional): Generate daily track plots if True

    Outputs
    -------
    Generates satellite track plots for multiple metrics:
    - NBIAS (normalized bias) tracks
    - NRMSE (normalized RMSE) tracks
    - BIAS (absolute bias) tracks
    - RMSE (absolute RMSE) tracks
    - MAE (mean absolute error) tracks

    If daily_tracks=True, separate files are generated for each day.

    Examples
    --------
    >>> main('conf.yaml', '20230901', '20230930')
    """
    # Load configuration
    conf = getConfigurationByID(conf_path, 'plot')
    outdir = conf.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)
    date = f"{start_date}_{end_date}"

    print("=" * 80)
    print(f"SATELLITE TRACK VALIDATION: {start_date} to {end_date}")
    print("=" * 80)

    # Get configuration parameters
    coast_resolution = getattr(conf, 'coast_resolution', 'i')
    daily_tracks = getattr(conf, 'daily_tracks', False)

    print(f"\nConfiguration:")
    print(f"  Coast resolution: {coast_resolution}")
    print(f"  Daily tracks: {daily_tracks}")

    # Process each experiment
    for dataset in conf.experiments:
        print(f"\n{'=' * 80}")
        print(f"Processing experiment: {dataset}")
        print(f"{'=' * 80}")

        # Load dataset
        ds_path = (conf.experiments[dataset].series).format(
            out_dir=outdir, date=date, experiment=dataset)
        ds_all = xr.open_dataset(ds_path).sel(model=dataset)

        print(f"\nDataset info:")
        print(f"  Total observations: {len(ds_all.obs)}")
        print(f"  Time range: {ds_all.time.min().values} to {ds_all.time.max().values}")

        sat_hs = ds_all.hs
        model_hs = ds_all.model_hs

        # Apply unbias correction if configured
        if hasattr(conf.filters, 'unbias') and \
                conf.filters.unbias in ['True', 'T', 'TRUE', 't']:
            print("\nApplying unbias correction...")
            sat_hs -= np.nanmean(sat_hs)
            sat_hs += np.nanmean(model_hs)

        print(f"\nNaN count: {np.sum(np.isnan(sat_hs))}")

        # Apply threshold filters
        if hasattr(conf.filters, 'max') and hasattr(conf.filters, 'min'):
            print(f"Applying threshold filters: [{conf.filters.min}, {conf.filters.max}] m")
            sat_hs = sat_hs.where(
                (ds_all.hs.values <= float(conf.filters.max)) &
                (ds_all.hs.values >= float(conf.filters.min)))
            model_hs = model_hs.where(
                (ds_all.model_hs.values <= float(conf.filters.max)) &
                (ds_all.model_hs.values >= float(conf.filters.min)))

        # Update dataset with filtered values
        ds_all.hs.values = sat_hs
        ds_all.model_hs.values = model_hs

        # Calculate validation metrics for tracks
        print("\nCalculating validation metrics...")
        ds_all['bias'] = model_hs - sat_hs
        ds_all['nbias'] = (model_hs - sat_hs) / sat_hs
        ds_all['rmse'] = np.sqrt((model_hs - sat_hs) ** 2)
        ds_all['nrmse'] = ds_all['rmse'] / sat_hs
        ds_all['mae'] = np.abs(model_hs - sat_hs)
        ds_all['nmae'] = ds_all['mae'] / sat_hs

        # Generate track plots
        print("\nGenerating satellite track plots...")
        variables = ['nbias', 'nrmse', 'bias', 'rmse', 'mae']

        for variable in variables:
            print(f"\nPlotting {variable} tracks...")
            outName = os.path.join(outdir, 'plots',
                                   f'tracks_{variable}_{dataset}_{date}.jpeg')
            plotTracks(ds_all, variable, coast_resolution, outName,
                       daily_track=daily_tracks)

    print(f"\n{'=' * 80}")
    print("SATELLITE TRACK VALIDATION COMPLETE")
    print(f"Output directory: {os.path.join(outdir, 'plots')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Wave Model Validation - Satellite Track Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--config', required=True,
                        help='Path to configuration YAML file')

    args = parser.parse_args()

    print("Note: This script is typically called from main.py")
