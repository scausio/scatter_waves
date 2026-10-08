"""
Satellite preprocessing: extract altimeter Hs inside the bounding box, apply
quality/land/outlier filters and merge all missions into a single file.

Output: {out_dir}/{start}_{end}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc
with variables hs, time, longitude, latitude, satellite along dimension 'obs'.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import netCDF4
import numpy as np
import xarray as xr
from xarray.coding.times import decode_cf_datetime

from utils import getConfigurationByID, daysBetweenDates, n_workers


def getFilenames(conf, day, sat_name):
    return os.path.join(conf.paths.sat, conf.filenames.sat.template).format(
        satName=sat_name, year=day[:4], satType=conf.sat_specifics.type,
        month=day[4:6], day=day)


def _read_var(nc, name, idx=None):
    var = nc.variables[name]
    data = var[:] if idx is None else var[idx]
    return np.ma.filled(np.ma.asarray(data, dtype='f8'), np.nan)


def read_track_in_box(path, conf):
    """Return the observations of one L3 file inside the bounding box (or None)."""
    bb = conf.processing.boundingBox
    spec = conf.sat_specifics
    filters = conf.processing.filters
    with netCDF4.Dataset(path) as nc:
        lat = _read_var(nc, spec.lat)
        lon = _read_var(nc, spec.lon)
        lon[lon > 180] -= 360
        inside = (lat > bb.ymin) & (lat < bb.ymax) & (lon > bb.xmin) & (lon < bb.xmax)
        if not inside.any():
            return None
        idx = np.flatnonzero(inside)
        hs = _read_var(nc, spec.hs, idx)
        tvar = nc.variables[spec.time]
        tvar.set_auto_maskandscale(False)
        times = decode_cf_datetime(tvar[idx], tvar.units, getattr(tvar, 'calendar', 'standard'))
        if filters.quality_check.variable_name:
            qc = _read_var(nc, filters.quality_check.variable_name, idx)
            hs[qc != filters.quality_check.value] = np.nan
        if filters.land_masking.variable_name:
            land = _read_var(nc, filters.land_masking.variable_name, idx)
            hs[land != filters.land_masking.value] = np.nan
    keep = np.isfinite(hs)
    return {'time': np.asarray(times, dtype='datetime64[ns]')[keep],
            'longitude': lon[idx][keep], 'latitude': lat[idx][keep], 'hs': hs[keep]}


def _process_sat_day(args):
    conf, sat_name, day = args
    files = sorted(glob(getFilenames(conf, day, sat_name)))
    chunks = [c for c in (read_track_in_box(f, conf) for f in files) if c is not None]
    if not chunks:
        return sat_name, day, len(files), None
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    # the filename glob also matches files ending at 00:00 of the next day:
    # keep only observations of this day, otherwise they are counted twice
    day0 = np.datetime64(f'{day[:4]}-{day[4:6]}-{day[6:]}', 'ns')
    in_day = (out['time'] >= day0) & (out['time'] < day0 + np.timedelta64(1, 'D'))
    return sat_name, day, len(files), {k: v[in_day] for k, v in out.items()}


def ocean_mask(shapefile, lon, lat):
    """True where the point falls inside the (sea) polygons of the shapefile."""
    import geopandas as gpd
    import shapely
    geom = shapely.union_all(gpd.read_file(shapefile).geometry.values)
    shapely.prepare(geom)
    return shapely.contains_xy(geom, lon, lat)


def zscore_mask(hs, sigma):
    """True for values within sigma standard deviations of the mean."""
    z = np.abs((hs - np.nanmean(hs)) / np.nanstd(hs))
    return z <= sigma


class Sat_processer:
    def __init__(self, conf_path, start_date, end_date, workers=None):
        self.conf = getConfigurationByID(conf_path, 'sat_preproc')
        self.days = daysBetweenDates(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date
        self.workers = n_workers(workers)
        os.makedirs(self.conf.paths.out_dir, exist_ok=True)

    @property
    def outname(self):
        return os.path.join(self.conf.paths.out_dir,
                            '{date}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc'.format(
                                date=f'{self.start_date}_{self.end_date}',
                                sigma=self.conf.processing.filters.zscore.sigma))

    def fingerprint(self):
        """Settings that determine the content of the output file."""
        c = self.conf
        return json.dumps({'sat_names': list(c.sat_names), 'paths': c.paths.sat,
                           'template': c.filenames.sat.template,
                           'specifics': dict(c.sat_specifics),
                           'bbox': dict(c.processing.boundingBox),
                           'filters': munch_to_dict(c.processing.filters),
                           'days': [self.days[0], self.days[-1]]}, sort_keys=True)

    def is_cached(self):
        if not os.path.exists(self.outname):
            return False
        with xr.open_dataset(self.outname) as ds:
            return ds.attrs.get('preproc_config') == self.fingerprint()

    def extract(self):
        tasks = [(self.conf, s, d) for s in self.conf.sat_names for d in self.days]
        t0 = time.time()
        if self.workers > 1:
            with ProcessPoolExecutor(self.workers) as ex:
                results = list(ex.map(_process_sat_day, tasks, chunksize=4))
        else:
            results = [_process_sat_day(t) for t in tasks]
        print(f'read {sum(r[2] for r in results)} files in {time.time() - t0:.1f} s '
              f'({self.workers} workers)')
        chunks, names = [], []
        for sat_name, day, nfiles, data in results:
            if nfiles == 0:
                print(f'  no files for {sat_name} {day}')
            if data is not None:
                chunks.append(data)
                names.append(np.full(len(data['hs']), sat_name))
        if not chunks:
            raise RuntimeError('No satellite observation found in the bounding box')
        merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        merged['satellite'] = np.concatenate(names)
        return merged

    def filter(self, data):
        filters = self.conf.processing.filters
        n0 = len(data['hs'])
        keep = np.ones(n0, bool)
        if filters.land_masking.shapefile:
            keep &= ocean_mask(filters.land_masking.shapefile, data['longitude'], data['latitude'])
            print(f'land mask: {n0 - keep.sum()} of {n0} obs removed')
        sigma = filters.zscore.sigma
        if sigma:
            zkeep = zscore_mask(data['hs'][keep], float(sigma))
            print(f'z-score ({sigma} sigma): {(~zkeep).sum()} obs removed')
            keep[np.flatnonzero(keep)[~zkeep]] = False
        return {k: v[keep] for k, v in data.items()}

    def to_dataset(self, data):
        order = np.argsort(data['time'], kind='stable')
        ds = xr.Dataset(
            {'hs': ('obs', data['hs'][order].astype('f4'),
                    {'units': 'm', 'long_name': 'satellite significant wave height'}),
             'satellite': ('obs', data['satellite'][order].astype(str))},
            coords={'time': ('obs', data['time'][order]),
                    'longitude': ('obs', data['longitude'][order]),
                    'latitude': ('obs', data['latitude'][order])})
        ds.attrs['preproc_config'] = self.fingerprint()
        return ds

    def run(self):
        print(self.outname)
        if self.is_cached():
            print('satellite file up to date, skipping')
            return self.outname
        ds = self.to_dataset(self.filter(self.extract()))
        for sat in self.conf.sat_names:
            print(f'  {sat}: {int((ds.satellite == sat).sum())} obs')
        ds.to_netcdf(self.outname)
        return self.outname


def munch_to_dict(m):
    return json.loads(json.dumps(m, default=str))
