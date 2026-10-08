"""
Spatial validation maps: pairs are binned on a regular lon/lat grid and the
metrics are computed per cell (vectorised with np.bincount).

Configuration (plot section):
  binning_resolution : cell size in degrees
  min_obs_per_cell   : cells with fewer pairs are masked (default 10)
  coast_resolution   : Basemap coastline resolution
  filters.percentile_thresholds : extra maps of bias/rmse for obs >= global percentile
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import xarray as xr

from mapping import MapCanvas, var_config
from pairs import load_pairs, load_plot_conf, plot_dir
from stats import grouped_metrics

MAP_VARIABLES = ['nobs', 'bias', 'nbias', 'rmse', 'nrmse', 'mae', 'si', 'correlation']


def grid_edges(values, step):
    """Cell edges aligned on multiples of step (cell centres at k * step)."""
    first = np.floor(np.nanmin(values) / step - 0.5) * step + step / 2
    last = np.ceil(np.nanmax(values) / step + 0.5) * step - step / 2
    return np.arange(first, last + step / 2, step)


def grid_metrics(pairs, step, min_obs=10, percentile_thresholds=None):
    """Dataset of metrics on (model, latitude, longitude)."""
    lon, lat = pairs.longitude.values, pairs.latitude.values
    lon_edges, lat_edges = grid_edges(lon, step), grid_edges(lat, step)
    nx, ny = len(lon_edges) - 1, len(lat_edges) - 1
    ix = np.clip(np.searchsorted(lon_edges, lon, side='right') - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(lat_edges, lat, side='right') - 1, 0, ny - 1)
    cell = iy * nx + ix
    obs = pairs.hs.values

    thresholds = {p: np.percentile(obs, p) for p in (percentile_thresholds or [])}
    per_model = []
    for model in pairs.model.values:
        mod = pairs.model_hs.sel(model=model).values
        res = grouped_metrics(mod, obs, cell, nx * ny, min_obs)
        for p, thr in thresholds.items():
            high = obs >= thr
            res_p = grouped_metrics(mod[high], obs[high], cell[high], nx * ny, max(min_obs // 2, 3))
            for key in ('bias', 'rmse', 'nbias', 'nobs'):
                res[f'{key}_p{p}'] = res_p[key]
        per_model.append(xr.Dataset({k: (('latitude', 'longitude'), v.reshape(ny, nx)) for k, v in res.items()}))
    out = xr.concat(per_model, dim='model').assign_coords(
        model=pairs.model.values,
        longitude=(lon_edges[:-1] + lon_edges[1:]) / 2,
        latitude=(lat_edges[:-1] + lat_edges[1:]) / 2)
    out.attrs.update(step=step, min_obs_per_cell=min_obs,
                     **{f'threshold_p{p}': float(t) for p, t in thresholds.items()})
    return out


def plotMap(canvas, grid, variable, outfile, title=''):
    values = grid[variable].values
    cfg = var_config(variable, values)
    data = values * cfg['scale']
    if variable.startswith('nobs'):
        data = np.where(data > 0, data, np.nan)
    fig, ax = canvas.new()
    step = grid.attrs['step']
    lon_e = np.append(grid.longitude.values - step / 2, grid.longitude.values[-1] + step / 2)
    lat_e = np.append(grid.latitude.values - step / 2, grid.latitude.values[-1] + step / 2)
    x, y = canvas.m(*np.meshgrid(lon_e, lat_e))
    vmax = cfg['vmax'] if cfg['vmax'] is not None else np.nanpercentile(data, 98)
    im = canvas.m.pcolormesh(x, y, np.ma.masked_invalid(data), cmap=cfg['cmap'],
                             vmin=cfg['vmin'], vmax=vmax, ax=ax, zorder=1)
    canvas.colorbar(fig, ax, im, cfg['title'], cfg['extend'])
    canvas.stats_box(ax, data, label='cells')
    ax.set_title(f"{cfg['title']}  {title}", loc='left', fontsize=13, fontweight='bold')
    canvas.save(outfile)


def main(conf_path, start_date, end_date):
    conf = load_plot_conf(conf_path)
    outdir = plot_dir(conf)
    date = f'{start_date}_{end_date}'
    step = float(conf.get('binning_resolution', 0.1))
    min_obs = int(conf.get('min_obs_per_cell', 10))
    percentiles = conf.filters.get('percentile_thresholds') or []

    pairs = load_pairs(conf, start_date, end_date)
    grid = grid_metrics(pairs, step, min_obs, percentiles)
    grid.to_netcdf(os.path.join(conf.out_dir.out_dir, f'spatial_metrics_{date}.nc'))

    canvas = MapCanvas(pairs.longitude.values, pairs.latitude.values, conf.get('coast_resolution', 'i'))
    variables = MAP_VARIABLES + [f'{k}_p{p}' for p in percentiles for k in ('bias', 'rmse')]
    for model in grid.model.values:
        g = grid.sel(model=model)
        for variable in variables:
            if variable == 'nobs' and model != grid.model.values[0]:
                continue  # common sample: the same for every model
            name = 'all' if variable == 'nobs' else model
            plotMap(canvas, g, variable, os.path.join(outdir, f'map_{variable}_{name}_{date}.jpeg'),
                    title='' if variable == 'nobs' else model)
        print(f'maps for {model} saved in {outdir}')
