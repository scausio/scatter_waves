"""
Model preprocessing: collocate model Hs on the satellite observations.

Spatial weights are computed once per model (the mesh does not change between
daily files); each daily file is then read once and only the needed nodes are
interpolated in time. The last record of the previous file is kept so that
observations between the last output of a day and midnight are interpolated
between the two files.

Interpolation (model_preproc.interpolation in the configuration):
  space: 'nearest' (default) - nearest node within filters.max_distance_in_space
         'linear'            - barycentric interpolation on the mesh triangles
                               (unstructured models only)
  time:  'nearest' (default) - nearest output within filters.max_distance_in_time
         'linear'            - linear between the two bracketing outputs, if they
                               are at most 2 * max_distance_in_time apart

Model types (model_preproc.datasets.models.<name>.type):
  unstr / regular : daily files, path with {day}
  series          : one dataset (zarr or netCDF) with hs(time, node) covering the
                    period, e.g. the WaveGraph test set; mesh coordinates from the
                    lat/lon variables, triangles from `mesh` (.gr3) or a 'tri' variable
  forecast        : WaveGraph forecasts, files hs(init_time, pred_step, node) matched
                    by the glob `path`, valid time = init_time + step_hours * pred_step.
                    One experiment per lead in `leads` (hours), named <name>_+<lead>h:
                    each observation is compared with the forecast issued at the
                    latest init such that its lead is in (lead - init spacing, lead].
                    Mesh as for 'series' (`coords` dataset with lat/lon, `mesh` .gr3)

Output: {out_dir}/{start}_{end}_series.nc with hs (obs) and model_hs (obs, model),
plus one {start}_{end}_{model}_series.nc cache per model entry.
"""
import json
import os
import time
from glob import glob

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from utils import getConfigurationByID, daysBetweenDates

UNSTRUCTURED = ('u', 'unstr', 'unstruct', 'unstructured')
REGULAR = ('cf', 'cf_compl', 'cf_compliant', 'reg', 'regular')
SERIES = ('series',)
FORECAST = ('forecast',)


def read_gr3(path):
    """Node lon, lat and 0-based triangles of a .gr3 mesh."""
    with open(path) as f:
        f.readline()
        n_elem, n_node = map(int, f.readline().split()[:2])
        nodes = np.loadtxt(f, max_rows=n_node)
        elems = np.loadtxt(f, max_rows=n_elem, dtype=np.int64)
    lon = nodes[:, 1].copy()
    lon[lon > 180] -= 360
    return lon, nodes[:, 2], elems[:, 2:5] - 1


def open_any(path):
    return xr.open_zarr(path) if path.rstrip('/').endswith('.zarr') else xr.open_dataset(path)


def mesh_coords(mc, ds=None):
    """lon, lat (float) and triangles of an unstructured mesh described in the config."""
    tri = None
    if mc.get('mesh'):
        lon, lat, tri = read_gr3(mc.mesh)
    if mc.get('coords') or ds is not None:
        src = open_any(mc.coords) if mc.get('coords') else ds
        lon = src[mc.lon].values.astype('f8')
        lat = src[mc.lat].values.astype('f8')
        lon = np.where(lon > 180, lon - 360, lon)
    if tri is None and ds is not None and 'tri' in ds:
        tri = ds['tri'].values
    return lon, lat, tri

_weights_cache = {}  # experiments on the same mesh share the spatial weights


def get_spatial_weights(mesh_lon, mesh_lat, x, y, method, max_dist, tri):
    key = (len(mesh_lon), float(mesh_lon.sum()), float(mesh_lat.sum()),
           None if tri is None else int(np.asarray(tri).sum()),
           method, max_dist, len(x), float(x.sum()), float(y.sum()))
    if key not in _weights_cache:
        _weights_cache[key] = SpatialWeights(mesh_lon, mesh_lat, x, y, method, max_dist, tri)
    return _weights_cache[key]


class SpatialWeights:
    """Nodes and weights to interpolate the model on fixed (lon, lat) points."""

    def __init__(self, mesh_lon, mesh_lat, x, y, method='nearest', max_dist=None, tri=None):
        self.n_mesh = len(mesh_lon)
        if method == 'linear' and tri is not None:
            self._linear(mesh_lon, mesh_lat, tri, x, y)
        else:
            if method == 'linear':
                print('  linear interpolation needs the mesh triangles: using nearest')
            self._nearest(mesh_lon, mesh_lat, x, y, max_dist)

    def _nearest(self, mesh_lon, mesh_lat, x, y, max_dist):
        dist, idx = cKDTree(np.column_stack((mesh_lon, mesh_lat))).query(np.column_stack((x, y)))
        self.valid = np.isfinite(dist)
        if max_dist is not None:
            self.valid &= dist <= float(max_dist)
        self.nodes = np.where(self.valid, idx, 0)[:, None]
        self.weights = np.ones_like(self.nodes, dtype='f8')

    def _linear(self, mesh_lon, mesh_lat, tri, x, y):
        from matplotlib.tri import Triangulation
        tri = np.asarray(tri, dtype=np.int64)
        if tri.min() == 1:  # WW3 connectivity is 1-based
            tri = tri - 1
        trifinder = Triangulation(mesh_lon, mesh_lat, tri).get_trifinder()
        elem = trifinder(x, y)
        self.valid = elem >= 0
        nodes = tri[np.where(self.valid, elem, 0)]
        x1, x2, x3 = (mesh_lon[nodes[:, i]] for i in range(3))
        y1, y2, y3 = (mesh_lat[nodes[:, i]] for i in range(3))
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        self.nodes = nodes
        self.weights = np.column_stack((l1, l2, 1 - l1 - l2))

    def apply(self, field):
        """field: (..., n_mesh) -> (..., n_points)"""
        return np.sum(field[..., self.nodes] * self.weights, axis=-1)


def time_weights(model_times, obs_times, method, max_hours):
    """Indices i0, i1 into model_times, weight of i1 and validity for each obs."""
    max_dt = np.timedelta64(int(float(max_hours) * 3600), 's')
    n = len(model_times)
    if method == 'linear':
        i0 = np.clip(np.searchsorted(model_times, obs_times, side='right') - 1, 0, n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
        t0, t1 = model_times[i0], model_times[i1]
        gap = (t1 - t0).astype('timedelta64[s]').astype('f8')
        alpha = np.where(gap > 0, (obs_times - t0).astype('timedelta64[s]').astype('f8') / np.where(gap > 0, gap, 1), 0.)
        valid = (obs_times >= model_times[0]) & (obs_times <= model_times[-1]) & (t1 - t0 <= 2 * max_dt)
        return i0, i1, alpha, valid
    i1 = np.clip(np.searchsorted(model_times, obs_times), 0, n - 1)
    i0 = np.maximum(i1 - 1, 0)
    nearest = np.where(np.abs(obs_times - model_times[i0]) <= np.abs(model_times[i1] - obs_times), i0, i1)
    valid = np.abs(model_times[nearest] - obs_times) <= max_dt
    return nearest, nearest, np.zeros(len(obs_times)), valid


class ModelReader:
    def __init__(self, conf, name):
        self.conf = conf
        self.name = name
        self.mc = conf.datasets.models[name]
        self.type = str(self.mc.type).lower()
        if self.type not in UNSTRUCTURED + REGULAR + SERIES:
            raise ValueError(f'model type {self.mc.type} not supported: use one of '
                             f'{UNSTRUCTURED + REGULAR + SERIES + FORECAST}')
        self._series = None
        interp = conf.get('interpolation') or {}
        self.space_method = interp.get('space', 'nearest')
        self.time_method = interp.get('time', 'nearest')

    def path(self, day):
        return self.mc.path.format(experiment=self.name, day=day)

    def read(self, path):
        """Return times and hs as (time, points) plus mesh lon, lat, triangles."""
        with xr.open_dataset(path) as ds:
            hs = ds[self.mc.hs]
            if self.type in REGULAR:
                hs = hs.stack(node=(self.mc.lat, self.mc.lon))
                hs = hs.isel(node=np.isfinite(hs.values).any(axis=hs.get_axis_num(self.mc.time)))
                lon = hs[self.mc.lon].values
                lat = hs[self.mc.lat].values
                tri = None
            else:
                lon = ds[self.mc.lon].values
                lat = ds[self.mc.lat].values
                tri = ds['tri'].values if 'tri' in ds else None
            times = ds[self.mc.time].values.astype('datetime64[ns]')
            values = hs.transpose(self.mc.time, ...).values
        return times, values, lon.astype('f8'), lat.astype('f8'), tri

    def read_day(self, day):
        """Model output of one day: (times, hs(time, points), lon, lat, tri) or None."""
        if self.type in SERIES:
            if self._series is None:
                ds = open_any(self.mc.path)
                self._series = (ds, *mesh_coords(self.mc, ds))
            ds, lon, lat, tri = self._series
            t0 = np.datetime64(f'{day[:4]}-{day[4:6]}-{day[6:]}', 'ns')
            sel = ds[self.mc.hs].sel({self.mc.time: slice(t0, t0 + np.timedelta64(1, 'D') - np.timedelta64(1, 's'))})
            if sel.sizes[self.mc.time] == 0:
                return None
            return sel[self.mc.time].values.astype('datetime64[ns]'), sel.transpose(self.mc.time, ...).values, lon, lat, tri
        path = self.path(day)
        if not os.path.exists(path):
            print(f'  {path} not found')
            return None
        return self.read(path)

    def collocate(self, sat, days):
        """Model Hs on each satellite observation (NaN where not available)."""
        obs_t = sat.time.values.astype('datetime64[ns]')
        x = sat.longitude.values.astype('f8')
        y = sat.latitude.values.astype('f8')
        if np.any(np.diff(obs_t) < np.timedelta64(0)):
            raise ValueError('satellite observations must be sorted in time')
        out = np.full(len(obs_t), np.nan, dtype='f4')
        weights, prev = None, None
        max_dt = self.conf.filters.max_distance_in_time
        for day in days:
            day_data = self.read_day(day)
            if day_data is None:
                prev = None
                continue
            times, values, lon, lat, tri = day_data
            if weights is None or weights.n_mesh != len(lon):
                t0 = time.time()
                weights = get_spatial_weights(lon, lat, x, y, self.space_method,
                                              self.conf.filters.max_distance_in_space, tri)
                print(f'  spatial weights ({self.space_method}): {weights.valid.sum()} of {len(x)} '
                      f'obs on the model domain [{time.time() - t0:.1f} s]')
            if prev is not None:
                times = np.concatenate((prev[0], times))
                values = np.concatenate((prev[1], values))
                start = np.searchsorted(obs_t, prev[0][0], side='right')
            else:
                start = np.searchsorted(obs_t, times[0] - np.timedelta64(int(float(max_dt) * 3600), 's'))
            stop = np.searchsorted(obs_t, times[-1], side='right')
            prev = (times[-1:], values[-1:])
            if stop <= start:
                continue
            sl = slice(start, stop)
            i0, i1, alpha, valid_t = time_weights(times, obs_t[sl], self.time_method, max_dt)
            valid = valid_t & weights.valid[sl]
            idx = np.flatnonzero(valid)
            if len(idx) == 0:
                continue
            nodes, w = weights.nodes[sl][idx], weights.weights[sl][idx]
            v0 = np.sum(values[i0[idx][:, None], nodes] * w, axis=1)
            v1 = np.sum(values[i1[idx][:, None], nodes] * w, axis=1)
            out[start + idx] = (1 - alpha[idx]) * v0 + alpha[idx] * v1
        return {self.name: out}

    def fingerprint(self, sat_path):
        st = os.stat(sat_path)
        return json.dumps({'model': dict(self.mc), 'sat': [sat_path, st.st_size, st.st_mtime],
                           'space': self.space_method, 'time': self.time_method,
                           'filters': dict(self.conf.filters)}, sort_keys=True, default=str)


class ForecastReader:
    """WaveGraph-like forecasts hs(init_time, pred_step, node), one experiment per lead."""

    def __init__(self, conf, name):
        self.conf = conf
        self.name = name
        self.mc = conf.datasets.models[name]
        interp = conf.get('interpolation') or {}
        self.space_method = interp.get('space', 'nearest')
        self.leads = [int(lead) for lead in self.mc.leads]
        self.step = np.timedelta64(int(float(self.mc.get('step_hours', 3)) * 3600), 's')
        self.init_dim = self.mc.get('init_time', 'init_time')
        self.step_dim = self.mc.get('pred_step', 'pred_step')
        self.files = sorted(glob(self.mc.path))
        if not self.files:
            raise FileNotFoundError(self.mc.path)
        self.names = [f'{name}_+{lead}h' for lead in self.leads]

    def index(self):
        inits, where = [], []
        for i, f in enumerate(self.files):
            with xr.open_dataset(f) as ds:
                for k, t in enumerate(ds[self.init_dim].values.astype('datetime64[ns]')):
                    inits.append(t)
                    where.append((i, k))
        order = np.argsort(inits)
        return np.array(inits)[order], np.array(where)[order]

    def collocate(self, sat, days=None):
        obs_t = sat.time.values.astype('datetime64[ns]')
        x = sat.longitude.values.astype('f8')
        y = sat.latitude.values.astype('f8')
        inits, where = self.index()
        spacing = np.median(np.diff(inits))
        print(f'  {len(inits)} forecasts in {len(self.files)} files, every '
              f'{spacing.astype("timedelta64[h]")}, leads {self.leads} h')
        lon, lat, tri = mesh_coords(self.mc)
        weights = get_spatial_weights(lon, lat, x, y, self.space_method,
                                      self.conf.filters.max_distance_in_space, tri)
        with xr.open_dataset(self.files[0]) as ds:
            steps = ds[self.step_dim].values.astype(int)
        step_pos = {s: i for i, s in enumerate(steps)}

        sel = {}  # lead name -> (obs, forecast index, s0, s1, alpha)
        for name, lead in zip(self.names, self.leads):
            lead_td = np.timedelta64(lead, 'h')
            j = np.minimum(np.searchsorted(inits, obs_t - lead_td, side='left'), len(inits) - 1)
            age = obs_t - inits[j]  # lead of the selected forecast at the obs time
            ok = (age <= lead_td) & (age > lead_td - spacing) & weights.valid
            frac = age.astype('timedelta64[s]').astype('f8') / self.step.astype('timedelta64[s]').astype('f8')
            s0 = np.floor(frac).astype(int)
            alpha = frac - s0
            s1 = np.where(alpha > 0, s0 + 1, s0)
            ok &= np.isin(s0, steps) & np.isin(s1, steps)
            idx = np.flatnonzero(ok)
            sel[name] = (idx, j[idx], s0[idx], s1[idx], alpha[idx])

        out = {name: np.full(len(obs_t), np.nan, dtype='f4') for name in self.names}
        file_of = where[:, 0]
        files_needed = np.unique(np.concatenate([file_of[v[1]] for v in sel.values()]))
        t0 = time.time()
        for n, fi in enumerate(files_needed):
            parts = {name: np.flatnonzero(file_of[v[1]] == fi) for name, v in sel.items()}
            needed = sorted({int(x) for name, k in parts.items() for x in np.r_[sel[name][2][k], sel[name][3][k]]})
            col = np.full(max(needed) + 1, -1)
            col[needed] = np.arange(len(needed))
            with xr.open_dataset(self.files[fi]) as ds:
                hs = ds[self.mc.hs].isel({self.step_dim: [step_pos[x] for x in needed]}) \
                    .transpose(self.init_dim, self.step_dim, ...).values
            for name, k in parts.items():
                if len(k) == 0:
                    continue
                obs, fc, a0, a1, alpha = (v[k] for v in sel[name])
                pos = where[fc, 1][:, None]
                nodes, w = weights.nodes[obs], weights.weights[obs]
                v0 = np.sum(hs[pos, col[a0][:, None], nodes] * w, axis=1)
                v1 = np.sum(hs[pos, col[a1][:, None], nodes] * w, axis=1)
                out[name][obs] = (1 - alpha) * v0 + alpha * v1
            if n % 50 == 0:
                print(f'  {n + 1}/{len(files_needed)} forecast files [{time.time() - t0:.0f} s]')
        return out

    def fingerprint(self, sat_path):
        st = os.stat(sat_path)
        return json.dumps({'model': dict(self.mc), 'sat': [sat_path, st.st_size, st.st_mtime],
                           'space': self.space_method, 'files': len(self.files),
                           'filters': dict(self.conf.filters)}, sort_keys=True, default=str)


def make_reader(conf, name):
    if str(conf.datasets.models[name].type).lower() in FORECAST:
        return ForecastReader(conf, name)
    return ModelReader(conf, name)


def collocation_days(start_date, end_date):
    """Days of model output needed: the period plus the next day for the last hour."""
    days = daysBetweenDates(start_date, end_date)
    return days + [_next_day(days[-1])]


def _next_day(day):
    return str(np.datetime64(f'{day[:4]}-{day[4:6]}-{day[6:]}') + 1).replace('-', '')


def submit(conf_path, start_date, end_date):
    conf_model = getConfigurationByID(conf_path, 'model_preproc')
    conf_sat = getConfigurationByID(conf_path, 'sat_preproc')
    date = f'{start_date}_{end_date}'
    outdir = conf_model.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)
    sat_path = conf_model.datasets.sat.path.format(
        out_dir=outdir, date=date, sigma=conf_sat.processing.filters.zscore.sigma)
    sat = xr.open_dataset(sat_path).load()
    days = collocation_days(start_date, end_date)

    columns = {}
    for name in conf_model.datasets.models:
        reader = make_reader(conf_model, name)
        cache = os.path.join(outdir, f'{date}_{name}_series.nc')
        fp = reader.fingerprint(sat_path)
        if os.path.exists(cache):
            with xr.open_dataset(cache) as c:
                if c.attrs.get('collocation_config') == fp and c.sizes['obs'] == sat.sizes['obs']:
                    print(f'{name}: using cached collocation {cache}')
                    for m in c.model.values:
                        columns[str(m)] = c['model_hs'].sel(model=m).values
                    continue
        print(f'{name}: collocating on {sat.sizes["obs"]} observations')
        t0 = time.time()
        result = reader.collocate(sat, days)
        for m, values in result.items():
            print(f'{m}: {int(np.isfinite(values).sum())} collocated obs')
        print(f'{name} done [{time.time() - t0:.1f} s]')
        xr.Dataset({'model_hs': (('model', 'obs'), np.stack(list(result.values())))},
                   coords={'model': list(result)}).assign_attrs(collocation_config=fp).to_netcdf(cache)
        columns.update(result)

    out = sat.copy()
    out['model_hs'] = xr.DataArray(np.stack(list(columns.values()), axis=1), dims=('obs', 'model'))
    out = out.assign_coords(model=list(columns))
    out['model_hs'].attrs = {'units': 'm', 'long_name': 'model significant wave height'}
    out.attrs['sat_file'] = os.path.basename(sat_path)
    outname = os.path.join(outdir, f'{date}_series.nc')
    out.to_netcdf(outname)
    print(f'{outname} saved')
    return outname
