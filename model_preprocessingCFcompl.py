import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID,checkOutdir
import numpy as np
from natsort import natsorted
from dask_jobqueue import LSFCluster
from dask.distributed import Client

def getModelIdxs(model,sat,conf,dataset):
    sat_idxs=get_satXYT(sat, conf)
    xm,ym,tm=get_modelXYT(model, conf, dataset)
    idx=[]
    for i,j,k in sat_idxs:

        nearestX=np.argmin(np.abs(xm - i))
        nearestY=np.argmin(np.abs(ym - j))
        nearestT=np.argmin(np.abs(tm - k.time))

        idx.append([nearestX, nearestY, nearestT])
        print ('time:',k, tm[nearestT])
        print ('x:', i, xm[nearestX])
        print ('y:', j,ym[nearestY])
    return idx

def get_modelXYT(ds, conf, dataset):
    return ds[conf.datasets.models[dataset].lon].values, ds[conf.datasets.models[dataset].lat].values, ds[conf.datasets.models[dataset].time]

def get_satXYT(ds, conf):
    return list(zip(ds[conf.datasets.sat.lon].values, ds[conf.datasets.sat.lat].values, ds[conf.datasets.sat.time]))

def checkOutDir(outPath):
    if not os.path.exists(outPath):
        os.makedirs(outPath)

def idx_main(model,sat,outdir,outname):
    checkOutDir(outdir)
    idx=getModelIdxs(model,sat)
    np.save(os.path.join(outdir,'%s_idx.npy'%outname),np.array(idx) )
    return idx

def getSeries(model,sat,conf,dataset,outname):
    #
    # ds=ds.isel(**{conf.datasets.models[dataset].lon:idxs[0],conf.datasets.models[dataset].lat:idxs[1],conf.datasets.models[dataset].time:idxs[2] }).compute()
    # ds.to_netcdf(os.path.join(outdir,'%s_series.nc'%outname))
    sat_idxs = get_satXYT(sat, conf)
    xm, ym, tm = get_modelXYT(model, conf, dataset)
    print (xm)

    #idx=[]
    series = []
    for i, j, k in sat_idxs:
        nearestX=np.argmin(np.abs(xm - i))
        nearestY=np.argmin(np.abs(ym - j))
        nearestT=np.argmin(np.abs(tm - k))
        #print (xm[nearestX],i, ym[nearestY],j,tm[nearestT],k)	
        dx=xm[nearestX]-i
        dy=ym[nearestY]-j
        dt=tm[nearestT]-k

        if (np.abs(dx)>float(conf.filters.max_distance_in_space)) or (np.abs(dy)>float(conf.filters.max_distance_in_space)) or ((np.abs(dt / np.timedelta64(1, 'h'))>conf.filters.max_distance_in_time)):
            series.append(np.nan)
            print ('skip point')
        else:
            subset=model[conf.datasets.models[dataset].hs].isel(**{conf.datasets.models[dataset].lon:nearestX,conf.datasets.models[dataset].lat:nearestY,conf.datasets.models[dataset].time:nearestT})

            series.append(subset.compute().data)
            # idx.append([nearestX,nearestY,nearestT])

            #print('time:', dt / np.timedelta64(1, 'h'))#k, tm[nearestT])
            #print('x:', dx)#i, xm[nearestX])
            #print('y:', dy)#)j, ym[nearestY])
    #series=np.clip(series,conf.filters.threshold.min,conf.filters.threshold.max)

    # idx=np.array(idx)
    # model[conf.datasets.models[dataset].hs].isel(
    #         **{conf.datasets.models[dataset].lon: idx[0], conf.datasets.models[dataset].lat: idx[1],
    #            conf.datasets.models[dataset].time: idx[2]})
    series=np.array(series)
    series[series <conf.filters.threshold.min]=np.nan
    series[series>conf.filters.threshold.max]=np.nan
    np.save('%s_series.npy'%outname,series)


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

    for dataset in conf.datasets.models:
        print (f'processing {dataset} dataset')
        outname=os.path.join(outdir,'{ds}_{yr}'.format(ds=dataset,yr=years))
        files_buffer=[]
        if years.find('_')>0:
            range_years=np.arange(int(years.split('_')[0]),int(years.split('_')[-1])+1)
            for year in range_years:	
                print (((conf.datasets.models[dataset].path).format(experiment=dataset,year=year)))
                [files_buffer.append(f) for f in glob((conf.datasets.models[dataset].path).format(experiment=dataset,year=year))]
        else:
            files_buffer.append( glob((conf.datasets.models[dataset].path).format(experiment=dataset,year=years)))
        print (files_buffer) 
        ds=xr.open_mfdataset(natsorted(files_buffer),combine='by_coords')
        # get timeseries
        #scatteredDs = client.scatter(ds)     # good

        #v=client.submit(getSeries,scatteredDs,sat,conf,dataset,outname)
        #v.result()
        getSeries(ds,sat,conf,dataset,outname)

if __name__ == '__main__':
    main()
