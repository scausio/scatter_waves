import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID
import numpy as np
from natsort import natsorted
from scipy.spatial import KDTree

def get_satXYT(ds, conf):
    return list(zip(ds[conf.datasets.sat.lon].values, ds[conf.datasets.sat.lat].values, ds[conf.datasets.sat.time]))

def getSeries(model,sat,conf,dataset,satname,outname):
    #define limited time from model and satellite

    first_time=max(sat.time.min(),model.time.min())
    last_time=min(sat.time.max(), model.time.max())
    sat=sat.isel(obs=(sat.time>=first_time)&(sat.time<=last_time))

    model = model.isel(time=(model.time>=first_time)&(model.time<=last_time))

    sat_idxs = np.array(get_satXYT(sat, conf))

    print (len(model.time), len(sat.time))
    print('get filtered time')
    time_idxs = np.array([np.argmin(np.abs(model.time - t)) for t in sat.time])
    time_filt=np.array([np.abs(((model.time[time_idxs[i]] - t).values / np.timedelta64(1, 'h'))) <= conf.filters.max_distance_in_time for i,t in enumerate(sat.time)]).astype(bool)

    print (len(time_filt))
    print (time_filt.shape)
    print(np.nansum(time_filt))
    print (model)
    print ('building kdtree')
    print (len(model[conf.datasets.models[dataset].lon].values[0]))
    tree = KDTree(np.array((model[conf.datasets.models[dataset].lon].values[0], model[conf.datasets.models[dataset].lat].values[0])).T)
    #
    print('getting nearest in space')
    sat_points=np.array((sat_idxs[time_filt,0], sat_idxs[time_filt,1])).T
    dist, idxs = tree.query(sat_points, k=1)
    print ('filtering in space')
    print (len(idxs))
    mask_dist=np.abs(dist) <= float(conf.filters.max_distance_in_space)
    print(len(idxs[mask_dist]))
    print ('slicing all')
    ds=model['hs'].values[time_idxs[time_filt][mask_dist],idxs[mask_dist]]
    print (ds.shape)
    print('slicing sat')
    sat=sat.isel(obs=time_filt).isel(obs=mask_dist)
    print (sat)
    print('replacing')
    sat['model_hs'].values=[ds]
    # sat['model_hs'].where(
    #     (sat['model_hs'] <= conf.filters.threshold.max) & (sat['model_hs'] >= conf.filters.threshold.min))
    # sat['hs'].where(
    #     (sat['hs'] <= conf.filters.threshold.max) & (sat['hs'] >= conf.filters.threshold.min))
    sat['hs'].attrs['satellite_file'] = satname
    sat.to_netcdf('%s_sat_series.nc' % outname)
    print ('%s_sat_series.nc saved' % outname)

def preprocesser(ds):
    return ds[['hs', 'time', 'node', 'longitude', 'latitude', 'tri']]

def main():
    conf = getConfigurationByID('.', 'model_preproc')
    conf_pre=getConfigurationByID('.','sat_preproc')

    years_name=f"{conf_pre.years[0]}_{conf_pre.years[-1]}" if len(conf_pre.years)>1  else conf_pre.years[0]

    outdir = conf.out_dir
    os.makedirs(outdir, exist_ok=True)

    sat_path=(conf.datasets.sat.path).format(year=years_name,sigma=conf_pre.processing.filters.zscore.sigma)
    sat=xr.open_dataset(sat_path)
    # sat.coords['model'] = np.array(list(conf.datasets.models.keys()), dtype=str)
    # print (sat)
    # model_variable = np.zeros_like(sat['hs'], shape=tuple(sat.sizes.values()))
    # sat['model_hs'] = xr.DataArray(model_variable, dims=sat.sizes.keys())
    for dataset in conf.datasets.models:
        outname=os.path.join(outdir,'{ds}_{yr}'.format(ds=dataset,yr=years_name))
        print ('processing %s dataset')
        filledPath=(conf.datasets.models[dataset].path).format(experiment=dataset,year='*')
        print ('searching for ', filledPath)
        ds=xr.open_mfdataset(natsorted(glob(filledPath)),preprocess=preprocesser)
        # get timeseries
        getSeries(ds,sat,conf,dataset,os.path.basename(sat_path),outname)

if __name__ == '__main__':
    main()
