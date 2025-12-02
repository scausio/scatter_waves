"""
Wave Model Validation - Spatial Map Analysis Module

This module generates spatial validation maps showing geographic distribution
of validation metrics. Useful for identifying regions where model performance
varies and for detecting spatial patterns in errors.

Features
--------
- 2D spatial binning of validation metrics
- Multiple metrics visualization (BIAS, RMSE, NRMSE, NBIAS)
- High-resolution coastline overlay
- Publication-quality geographic maps
- Percentile-based spatial validation

Author: Salvatore Causio
Date: 2025
"""

import matplotlib

matplotlib.use('Agg')
from stats import metrics
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import getConfigurationByID, ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings('ignore')


def coords2bins(ds_all: xr.Dataset, x: np.ndarray, y: np.ndarray,
                step: float) -> tuple:
    """
    Determine bin edges for spatial gridding based on data extent.

    Parameters
    ----------
    ds_all : xr.Dataset
        Dataset containing latitude and longitude variables
    x : np.ndarray
        Global longitude grid
    y : np.ndarray
        Global latitude grid
    step : float
        Grid spacing in degrees

    Returns
    -------
    tuple
        (lon_bins, lat_bins) - Arrays of bin edges for longitude and latitude
    """
    y_min = np.nanmin(ds_all.latitude.values) - step
    y_max = np.nanmax(ds_all.latitude.values) + step
    x_min = np.nanmin(ds_all.longitude.values) - step
    x_max = np.nanmax(ds_all.longitude.values) + step

    lon_bins = np.arange(x[np.argmin(np.abs(x - x_min))],
                         x[np.argmin(np.abs(x - x_max))] + step, step)
    lat_bins = np.arange(y[np.argmin(np.abs(y - y_min))],
                         y[np.argmin(np.abs(y - y_max))] + step, step)

    return lon_bins, lat_bins


def targetGrid(step: float) -> tuple:
    """
    Create global target grid for binning.

    Parameters
    ----------
    step : float
        Grid spacing in degrees

    Returns
    -------
    tuple
        (x, y) - Global longitude and latitude arrays
    """
    x = np.arange(-180 - (step / 2), 180 + step, step)
    y = np.arange(-90 - (step / 2), 90 + step, step)
    return x, y


def get2Dbins(data: xr.Dataset, step: float,
              percentile_thresholds: list = None) -> xr.Dataset:
    """
    Perform 2D spatial binning and calculate metrics for each grid cell.

    This function bins scattered observations into a regular grid and
    computes validation metrics for each cell. Useful for creating
    spatial maps of model performance.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing model_hs, hs, latitude, longitude
    step : float
        Grid spacing in degrees (e.g., 0.05 for ~5km resolution)
    percentile_thresholds : list, optional
        Percentiles for extreme value validation

    Returns
    -------
    xr.Dataset
        Gridded dataset with computed metrics at each grid cell

    Notes
    -----
    The function displays progress as it processes each longitude bin.
    For large datasets, this can take several minutes.
    """
    print(f'Creating target grid at {step}° resolution')
    x, y = targetGrid(step)

    data = data.sortby('latitude')
    print('Starting 2D spatial binning...')

    lon_bins, lat_bins = coords2bins(data, x, y, step)

    # Group by longitude bins
    grouped_lon = data.groupby_bins('longitude', lon_bins)
    print(f'Processing {len(grouped_lon)} longitude bins...')

    lon_ = [i.mid for i in grouped_lon.groups.keys()]
    buffer = []

    # Process each longitude bin
    for i, lon_group in enumerate(grouped_lon._iter_grouped()):
        remaining = len(list(grouped_lon._iter_grouped())) - i
        if remaining % 10 == 0 or remaining <= 5:
            print(f'  {remaining} longitude bins remaining...')

        # Group by latitude bins within this longitude
        lat_group = lon_group.groupby_bins('latitude', lat_bins)
        lat_ = [i.mid for i in lat_group.groups.keys()]

        # Calculate metrics for each grid cell
        if percentile_thresholds:
            results = lat_group.apply(lambda x: metrics(x, percentile_thresholds))
        else:
            results = lat_group.apply(metrics)

        # Rename and assign coordinates
        results = results.rename(latitude_bins='latitude')
        results = results.assign_coords({"latitude": [i.mid for i in results.latitude.values]})
        buffer.append(results)

    print('Assembling final grid...')
    out = xr.concat([buffer[i] for i in np.argsort(lon_)], 'longitude')
    out = out.assign_coords({"longitude": np.array(lon_)[np.argsort(lon_)]})

    print('✓ Spatial binning complete')
    return out.transpose()


def plotMap(ds: xr.Dataset, variable: str, coast_resolution: str,
            outfile: str, title_suffix: str = '') -> None:
    """
    Create publication-quality spatial validation map.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded dataset with computed metrics
    variable : str
        Variable to plot ('bias', 'rmse', 'nrmse', 'nbias', 'mae', etc.)
    coast_resolution : str
        Basemap coastline resolution ('c', 'l', 'i', 'h', 'f')
        c=crude, l=low, i=intermediate, h=high, f=full
    outfile : str
        Path for output file
    title_suffix : str, optional
        Additional text for title (e.g., for percentile info)

    Notes
    -----
    Color scales and ranges are optimized for each metric type:
    - Bias metrics use diverging colormaps (blue-white-red)
    - Error metrics use sequential colormaps (viridis, jet)
    - Percentile-based metrics use enhanced ranges
    """
    print(f'Generating map for {variable}...')

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

    # Draw parallels and meridians
    tick_spacing = ticker(lat_min, lat_max)
    meridians = np.arange(-180, 180 + tick_spacing, tick_spacing)
    parallels = np.arange(-90, 90 + tick_spacing, tick_spacing)
    m.drawparallels(parallels, labels=[True, False, False, True],
                    linewidth=0.5, fontsize=10, color='gray')
    m.drawmeridians(meridians, labels=[True, False, False, True],
                    linewidth=0.5, fontsize=10, color='gray')

    # Configure variable-specific settings
    var_config = {
        'bias': {
            'title': 'Bias [m]',
            'vmin': -1.0,
            'vmax': 1.0,
            'cmap': 'RdBu_r',
            'extend': 'both'
        },
        'rmse': {
            'title': 'RMSE [m]',
            'vmin': 0,
            'vmax': 1.0,
            'cmap': 'YlOrRd',
            'extend': 'max'
        },
        'mae': {
            'title': 'MAE [m]',
            'vmin': 0,
            'vmax': 0.8,
            'cmap': 'YlOrRd',
            'extend': 'max'
        },
        'nrmse': {
            'title': f'NRMSE [%]',
            'vmin': 0,
            'vmax': 60,
            'cmap': 'viridis',
            'extend': 'max'
        },
        'nbias': {
            'title': f'NBIAS [%]',
            'vmin': -60,
            'vmax': 60,
            'cmap': 'RdBu_r',
            'extend': 'both'
        },
        'correlation': {
            'title': 'Correlation',
            'vmin': 0,
            'vmax': 1,
            'cmap': 'RdYlGn',
            'extend': 'neither'
        },
        'skill_score': {
            'title': 'Skill Score',
            'vmin': 0,
            'vmax': 1,
            'cmap': 'RdYlGn',
            'extend': 'neither'
        }
    }

    # Get configuration or use defaults
    if variable in var_config:
        config = var_config[variable]
    else:
        # Default configuration for unknown variables
        config = {
            'title': variable.upper(),
            'vmin': np.nanpercentile(ds[variable], 5),
            'vmax': np.nanpercentile(ds[variable], 95),
            'cmap': 'viridis',
            'extend': 'both'
        }

    # Apply scaling for normalized metrics
    if variable in ['nrmse', 'nbias']:
        var_data = ds[variable] * 100
    else:
        var_data = ds[variable]

    # Calculate mean for title
    mean_val = float(np.nanmean(var_data))

    # Create title with mean value
    title = f"{config['title']}: μ = {mean_val:.3f}"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title, loc='left', fontsize=14, fontweight='bold', pad=15)

    # Plot data
    lon_grid, lat_grid = np.meshgrid(ds.longitude, ds.latitude)
    x, y = m(lon_grid, lat_grid)

    im = m.pcolormesh(x, y, var_data.T,
                      cmap=config['cmap'],
                      vmin=config['vmin'],
                      vmax=config['vmax'],
                      shading='auto',
                      alpha=0.8)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, extend=config['extend'])
    cbar.set_label(config['title'], fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    # Add statistics textbox
    stats_text = f"N = {int(np.sum(~np.isnan(var_data)))} cells\n"
    stats_text += f"μ = {mean_val:.3f}\n"
    stats_text += f"σ = {float(np.nanstd(var_data)):.3f}\n"
    stats_text += f"min = {float(np.nanmin(var_data)):.3f}\n"
    stats_text += f"max = {float(np.nanmax(var_data)):.3f}"

    # Add textbox to plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props, family='monospace')

    # Save figure
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ Map saved: {outfile}')


def main(conf_path: str, start_date: str, end_date: str) -> None:
    """
    Main function for spatial validation map generation.

    Creates geographic maps showing spatial distribution of validation metrics.
    Useful for identifying regions with systematic model biases or high errors.

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
    - binning_resolution: Grid spacing in degrees (e.g., 0.05)
    - coast_resolution: Coastline detail level ('c', 'l', 'i', 'h', 'f')
    - filters: Quality control parameters
    - percentile_thresholds (optional): For extreme value maps

    Outputs
    -------
    Generates spatial maps for multiple metrics:
    - NBIAS (normalized bias)
    - NRMSE (normalized RMSE)
    - MAE (mean absolute error)
    - BIAS (absolute bias)
    - RMSE (root mean square error)
    - Correlation (if available)
    - Percentile-based metrics (if configured)

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
    print(f"SPATIAL MAP VALIDATION: {start_date} to {end_date}")
    print("=" * 80)

    # Get configuration parameters
    binning_resolution = getattr(conf, 'binning_resolution', 0.05)
    coast_resolution = getattr(conf, 'coast_resolution', 'i')
    percentile_thresholds = getattr(conf.filters, 'percentile_thresholds', None)

    print(f"\nConfiguration:")
    print(f"  Binning resolution: {binning_resolution}°")
    print(f"  Coast resolution: {coast_resolution}")
    if percentile_thresholds:
        print(f"  Percentile thresholds: {percentile_thresholds}")

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
        print(f"  Latitude range: [{float(ds_all.latitude.min()):.2f}, "
              f"{float(ds_all.latitude.max()):.2f}]")
        print(f"  Longitude range: [{float(ds_all.longitude.min()):.2f}, "
              f"{float(ds_all.longitude.max()):.2f}]")

        sat_hs = ds_all.hs
        model_hs = ds_all.model_hs

        # Apply unbias correction if configured
        if hasattr(conf.filters, 'unbias') and \
                conf.filters.unbias in ['True', 'T', 'TRUE', 't']:
            print("\nApplying unbias correction...")
            sat_hs -= np.nanmean(sat_hs)
            sat_hs += np.nanmean(model_hs)

        print(f"\nNaN count in satellite data: {np.sum(np.isnan(sat_hs))}")

        # Perform 2D binning
        print(f"\nPerforming spatial binning...")
        ds_binned = get2Dbins(ds_all, binning_resolution, percentile_thresholds)

        print(f"\nBinned grid dimensions:")
        print(f"  Longitude bins: {len(ds_binned.longitude)}")
        print(f"  Latitude bins: {len(ds_binned.latitude)}")

        # Save binned dataset
        print('\nSaving binned dataset...')
        binned_path = os.path.join(outdir, f'spatial_metrics_{dataset}_{date}.nc')
        ds_binned.to_netcdf(binned_path)
        print(f'  ✓ Saved: {binned_path}')

        # Generate maps for standard metrics
        print('\nGenerating spatial validation maps...')
        standard_vars = ['nbias', 'nrmse', 'mae', 'bias', 'rmse']

        # Add correlation if available
        if 'correlation' in ds_binned:
            standard_vars.append('correlation')

        for variable in standard_vars:
            if variable in ds_binned:
                outName = os.path.join(outdir, 'plots',
                                       f'map_{variable}_{dataset}_{date}.jpeg')
                plotMap(ds_binned, variable, coast_resolution, outName)

        # Generate percentile-based maps if configured
        if percentile_thresholds:
            print('\nGenerating percentile-based validation maps...')
            for perc in percentile_thresholds:
                for metric in ['bias', 'rmse', 'mae']:
                    var_name = f'{metric}_p{perc}'
                    if var_name in ds_binned:
                        outName = os.path.join(outdir, 'plots',
                                               f'map_{var_name}_{dataset}_{date}.jpeg')
                        plotMap(ds_binned, var_name, coast_resolution, outName,
                                title_suffix=f'(P{perc})')

    print(f"\n{'=' * 80}")
    print("SPATIAL MAP VALIDATION COMPLETE")
    print(f"Output directory: {os.path.join(outdir, 'plots')}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Wave Model Validation - Spatial Map Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--config', required=True,
                        help='Path to configuration YAML file')

    args = parser.parse_args()

    print("Note: This script is typically called from main.py")
