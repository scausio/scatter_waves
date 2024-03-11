import xarray as xr
import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID,daysBetweenDates
import numpy as np
from natsort import natsorted
from scipy.spatial import KDTree

def get_satXYT(ds, conf):
    return list(zip(ds[conf.sat_specifics.lon].values, ds[conf.sat_specifics.lat].values, ds[conf.sat_specifics.time]))

def getSeries(model,sat,conf,conf_sat,dataset,satname,outname):
    #define limited time from model and satellite
    first_time=max(sat.time.min(),model.time.min())
    last_time=min(sat.time.max(), model.time.max())
    sat=sat.isel(obs=(sat.time>=first_time)&(sat.time<=last_time))
    model = model.isel(time=(model.time>=first_time)&(model.time<=last_time))
    print('get filtered time')
    time_idxs = np.array([np.argmin(np.abs(model.time.values - t.values)) for t in sat.time])
    time_filt=np.array([np.abs(((model.time[time_idxs[i]] - t).values / np.timedelta64(1, 'h'))) <= conf.filters.max_distance_in_time for i,t in enumerate(sat.time)]).astype(bool)
    sat_idxs = np.array(get_satXYT(sat, conf_sat))

    print ('building kdtree')
    print (len(model[conf.datasets.models[dataset].lon].values))
    tree = KDTree(np.array((model[conf.datasets.models[dataset].lon].values, model[conf.datasets.models[dataset].lat].values)).T)
    #
    print('getting nearest in space')
    print (np.array(sat_idxs).shape)
    try:
        sat_points=np.array((sat_idxs[time_filt,0], sat_idxs[time_filt,1])).T
    except:
        return
    dist, idxs = tree.query(sat_points, k=1)
    print ('filtering in space')

    mask_dist=np.abs(dist) <= float(conf.filters.max_distance_in_space)

    print ('slicing all')
    ds=model['hs'].values[time_idxs[time_filt][mask_dist],idxs[mask_dist]]

    print('slicing sat')
    sat=sat.isel(obs=time_filt).isel(obs=mask_dist)
    print('replacing')
    sat['model_hs'].values=np.array([ds]).T
    sat['hs'].attrs['satellite_file'] = satname
    sat.to_netcdf('%s' % outname)
    print ('%s saved' % outname)
    return sat

def preprocesser(ds):
    return ds[['hs', 'time', 'node', 'longitude', 'latitude', 'tri']]

def submit(conf_path,start_date,end_date):
    conf_model = getConfigurationByID(conf_path, 'model_preproc')
    conf_sat=getConfigurationByID(conf_path,'sat_preproc')
    date = f"{start_date}_{end_date}"
    #years=f"{conf_sat.years[0]}_{conf_sat.years[-1]}" if len(conf_sat.years)>1  else conf_sat.years[0]

    outdir = conf_model.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)
    os.path.join(outdir,'{date}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc')
    sat_path=(conf_model.datasets.sat.path).format(out_dir=outdir,date=date,sigma=conf_sat.processing.filters.zscore.sigma)
    sat=xr.open_dataset(sat_path)


    for dataset in conf_model.datasets.models:
        outname=os.path.join(outdir,'{date}_{ds}_series.nc'.format(ds=dataset,date=date))
        if not os.path.exists(outname):
            buffer = []
            for day in daysBetweenDates(start_date,end_date):
                print (f'extracting day {day} from {dataset}')
                outname_day = os.path.join(outdir, '{day}_{ds}.nc'.format(ds=dataset, day=day))
                print (outname_day)
                if not os.path.exists(outname_day):
                    print (f'processing {dataset}')
                    filledPath=(conf_model.datasets.models[dataset].path).format(experiment=dataset,day=day)
                    print ('searching for ', filledPath)
                    ds = preprocesser(xr.open_dataset(filledPath))

                    daily_ds = getSeries(ds, sat, conf_model,conf_sat, dataset, os.path.basename(sat_path), outname_day)
                    if daily_ds:
                        buffer.append(daily_ds)
                else:
                    buffer.append(xr.open_dataset(outname_day))
            xr.concat(buffer,dim='obs').to_netcdf(outname)


