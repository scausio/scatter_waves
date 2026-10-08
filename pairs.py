"""
Load the collocated satellite/model pairs and apply the plot filters once, so
that scatter plots, maps, time series and tracks are computed on the same sample.
"""
import os

import numpy as np
import xarray as xr

from utils import as_bool, getConfigurationByID


def load_pairs(conf, start_date, end_date, verbose=True):
    """
    Return a Dataset with hs (obs), model_hs (obs, model) and coordinates,
    restricted to the observations valid for the satellite and for every
    experiment listed in plot.experiments.

    Filters (plot.filters):
      min, max : valid Hs range, applied to satellite and models
      ntimes   : drop pairs where |model - sat| > ntimes * model
      unbias   : remove the mean bias of each model (model - <model> + <sat>)
    """
    date = f'{start_date}_{end_date}'
    if conf.get('experiments'):
        experiments = list(conf.experiments)
        paths = {name: conf.experiments[name].series.format(
            out_dir=conf.out_dir.out_dir, date=date, experiment=name) for name in experiments}
    else:  # all the experiments of the series file
        path = conf.get('series', '{out_dir}/{date}_series.nc').format(out_dir=conf.out_dir.out_dir, date=date)
        with xr.open_dataset(path) as ds:
            experiments = [str(m) for m in ds.model.values]
        paths = {name: path for name in experiments}

    columns, base = [], None
    for path in dict.fromkeys(paths.values()):
        names = [n for n in experiments if paths[n] == path]
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        if base is None:
            base = ds.drop_vars('model_hs').drop_dims('model', errors='ignore')
        elif ds.sizes['obs'] != base.sizes['obs']:
            raise ValueError(f'{path} has different observations from the other series files')
        columns.append(ds['model_hs'].sel(model=names))
    model_hs = xr.concat(columns, dim='model').sel(model=experiments).transpose('obs', 'model')

    filters = conf.filters
    sat = base['hs'].values
    mod = model_hs.values
    valid = np.isfinite(sat) & np.all(np.isfinite(mod), axis=1)
    report = [('collocated', valid.sum())]
    if 'min' in filters and 'max' in filters:
        lo, hi = float(filters.min), float(filters.max)
        in_range = (sat >= lo) & (sat <= hi) & np.all((mod >= lo) & (mod <= hi), axis=1)
        valid &= in_range
        report.append((f'Hs in [{lo}, {hi}] m', valid.sum()))
    if filters.get('ntimes'):
        n = float(filters.ntimes)
        with np.errstate(invalid='ignore'):
            outlier = np.any(np.abs(mod - sat[:, None]) > n * mod, axis=1)
        valid &= ~outlier
        report.append((f'|model-sat| <= {n} x model', valid.sum()))

    pairs = base.isel(obs=valid)
    pairs['model_hs'] = model_hs.isel(obs=valid)
    if as_bool(filters.get('unbias', False)):
        shift = pairs['model_hs'].mean('obs') - pairs['hs'].mean('obs')
        pairs['model_hs'] = pairs['model_hs'] - shift
        pairs.attrs['unbiased'] = 'model mean bias removed: ' + ', '.join(
            f'{m}={float(s):+.3f} m' for m, s in zip(experiments, shift.values))
        report.append(('unbias applied', valid.sum()))
    if 'satellite' not in pairs:
        pairs['satellite'] = ('obs', np.full(pairs.sizes['obs'], 'ALL'))

    if verbose:
        print(f'pairs ({", ".join(experiments)}): {len(sat)} observations in series file')
        for label, n in report:
            print(f'  {label:<32s} {n:8d}')
    if pairs.sizes['obs'] == 0:
        raise ValueError('No valid satellite/model pair after filtering')
    return pairs


def plot_dir(conf):
    out = os.path.join(conf.out_dir.out_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    return out


def load_plot_conf(conf_path):
    return getConfigurationByID(conf_path, 'plot')
