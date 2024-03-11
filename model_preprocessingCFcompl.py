import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID,checkOutdir
import numpy as np
from natsort import natsorted
from dask_jobqueue import LSFCluster
from dask.distributed import Client
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
    x=ds.longitude.values
    y=ds.latitude.values
    xx,yy=np.meshgrid(x,y)
    land=np.isnan(ds.hs.isel(time=0).values)
    x_sea=xx[~land]
    y_sea = yy[~land]

    hs_sea=ds.hs.values[:,~land]

    out=xr.Dataset()
    out['time']=ds.time
    out['node']=range(len(x_sea))
    out['longitude']=('node',x_sea)
    out['latitude'] = ('node', y_sea)
    out['hs'] = (('time','node'), hs_sea)

    return out

def main():
    #cluster = LSFCluster(cores=4,
    #                 processes=2,
    #                 memory="10GB",
    #                 project='0512',
    #                 queue='p_medium',
    #                 interface='ib0')#,
                     #scheduler_options={"host":"172.25.2.199","dashboard_address": ":8787"})
    #cluster.scale(144)  # Start 50 workers in 50 jobs that match the description above
    #client = Client(cluster)    # Connect to that cluster
    
    conf_pre = getConfigurationByID('.', 'sat_preproc')
    conf = getConfigurationByID('.', 'model_preproc')
    checkOutdir(conf.out_dir)
    years=f"{conf.years[0]}_{conf.years[-1]}" if len(conf.years)>1  else conf.years[0]
    outdir=conf.out_dir
    sat_path=(conf.datasets.sat.path).format(year=years,sigma=conf_pre.processing.filters.zscore.sigma)
    sat=xr.open_dataset(sat_path)
    # sat.coords['model'] = np.array(list(conf.datasets.models.keys()), dtype=str)
    # print (sat)
    # model_variable = np.zeros_like(sat['hs'], shape=tuple(sat.sizes.values()))
    # sat['model_hs'] = xr.DataArray(model_variable, dims=sat.sizes.keys())
    for dataset in conf.datasets.models:
        print (f'processing {dataset} dataset')
        outname=os.path.join(outdir,'{ds}_{yr}'.format(ds=dataset,yr=years))
        files_buffer=[]
        if years.find('_')>0:
            range_years=np.arange(int(years.split('_')[0]),int(years.split('_')[-1])+1)
            for year in range_years:
                print (year)
                print (((conf.datasets.models[dataset].path).format(experiment=dataset,year=year)))
                [files_buffer.append(f) for f in glob((conf.datasets.models[dataset].path).format(experiment=dataset,year=year))]
        else:
            files_buffer.append( glob((conf.datasets.models[dataset].path).format(experiment=dataset,year=years)))

        ds=xr.open_mfdataset(natsorted(files_buffer),preprocess=preprocesser,combine='by_coords')
        print (ds)
        # get timeseries
        #scatteredDs = client.scatter(ds)     # good

        #v=client.submit(getSeries,scatteredDs,sat,conf,dataset,outname)
        #v.result()
        getSeries(ds,sat,conf,dataset,os.path.basename(sat_path),outname)

if __name__ == '__main__':
    main()
