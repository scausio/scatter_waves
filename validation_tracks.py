"""
Along-track validation: satellite ground tracks coloured by the pointwise error.

Configuration (plot section):
  coast_resolution : Basemap coastline resolution
  daily_tracks     : also save one map per day (default False)
"""
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from mapping import MapCanvas, var_config
from pairs import load_pairs, load_plot_conf, plot_dir
from utils import as_bool

TRACK_VARIABLES = ['bias', 'nbias', 'abs_error']


def pointwise(model_hs, sat_hs):
    diff = model_hs - sat_hs
    return {'bias': diff, 'nbias': diff / sat_hs, 'abs_error': np.abs(diff)}


def plotTracks(canvas, lon, lat, values, variable, outfile, title=''):
    cfg = var_config(variable, values)
    data = values * cfg['scale']
    fig, ax = canvas.new()
    x, y = canvas.m(lon, lat)
    order = np.argsort(np.abs(np.nan_to_num(data)))  # largest errors on top
    im = ax.scatter(x[order], y[order], c=data[order], cmap=cfg['cmap'], vmin=cfg['vmin'],
                    vmax=cfg['vmax'], s=14, edgecolors='none', zorder=4)
    canvas.colorbar(fig, ax, im, cfg['title'], cfg['extend'])
    canvas.stats_box(ax, data)
    ax.set_title(f"{cfg['title']}  {title}", loc='left', fontsize=13, fontweight='bold')
    canvas.save(outfile)


def main(conf_path, start_date, end_date):
    conf = load_plot_conf(conf_path)
    outdir = plot_dir(conf)
    date = f'{start_date}_{end_date}'
    daily = as_bool(conf.get('daily_tracks', False))

    pairs = load_pairs(conf, start_date, end_date, verbose=False)
    lon, lat, sat = pairs.longitude.values, pairs.latitude.values, pairs.hs.values
    days = pd.DatetimeIndex(pairs.time.values).floor('D')
    canvas = MapCanvas(lon, lat, conf.get('coast_resolution', 'i'))

    for model in pairs.model.values:
        errors = pointwise(pairs.model_hs.sel(model=model).values, sat)
        for variable in TRACK_VARIABLES:
            outfile = os.path.join(outdir, f'tracks_{variable}_{model}_{date}.jpeg')
            plotTracks(canvas, lon, lat, errors[variable], variable, outfile, model)
            if daily:
                for day in days.unique():
                    sel = days == day
                    plotTracks(canvas, lon[sel], lat[sel], errors[variable][sel], variable,
                               outfile.replace('.jpeg', f'_{day:%Y%m%d}.jpeg'), f'{model} {day:%Y-%m-%d}')
        print(f'track maps for {model} saved in {outdir}')
