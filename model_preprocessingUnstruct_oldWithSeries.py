import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID
import numpy as np
from natsort import natsorted

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
    return ds[conf.datasets.models[dataset].lon].isel(time=0).values, ds[conf.datasets.models[dataset].lat].isel(time=0).values, ds[conf.datasets.models[dataset].time]

def get_satXYT(ds, conf):
    return list(zip(ds[conf.datasets.sat.lon].values, ds[conf.datasets.sat.lat].values, ds[conf.datasets.sat.time]))

def idx_main(model,sat,outdir,outname):
    os.makedirs(outdir,exist_ok=True)
    idx=getModelIdxs(model,sat)
    np.save(os.path.join(outdir,'%s_idx.npy'%outname),np.array(idx) )
    return idx

def getSeries(model,sat,conf,dataset,outname):
    #
    # ds=ds.isel(**{conf.datasets.models[dataset].lon:idxs[0],conf.datasets.models[dataset].lat:idxs[1],conf.datasets.models[dataset].time:idxs[2] }).compute()
    # ds.to_netcdf(os.path.join(outdir,'%s_series.nc'%outname))
    sat_idxs = get_satXYT(sat, conf)
    xm, ym, tm = get_modelXYT(model, conf, dataset)
    series = []
    for i, j, k in sat_idxs:
        ds=model.sel(time=k,method='nearest')
        if np.abs((ds.time-k) / np.timedelta64(1, 'h'))>conf.filters.max_distance_in_time:
            print ('skip point')
            series.append(np.nan)
            continue
        else:
            pass
        #ds[conf.datasets.models[dataset].lon], ds[conf.datasets.models[dataset].lat], ds[conf.datasets.models[dataset].time]
        dist=np.abs(xm - i)+np.abs(ym - j)

        nearestNode=np.argmin(dist)
        ds=ds.isel(node=nearestNode)
        #nearestX=np.argmin(np.abs(ds.longitude - i))
        #nearestY=np.argmin(np.abs(ds.latitude - j))
        #nearestT=np.argmin(np.abs(tm - k.time))
        #dx=xm[nearestX]-i
        #dy=ym[nearestY]-j
        #dt=tm[nearestT]-k

        if np.abs(dist[nearestNode])>float(conf.filters.max_distance_in_space) :
            series.append(np.nan)
            print ('skip point')
        else:
            subset=ds[conf.datasets.models[dataset].hs]#.isel(**{conf.datasets.models[dataset].lon:nearestX,conf.datasets.models[dataset].lat:nearestY,conf.datasets.models[dataset].time:nearestT})

            series.append(subset.compute().values)
            # idx.append([nearestX,nearestY,nearestT])


            print('time:', ds.time, k)#k, tm[nearestT])
            print('coords:', ds.longitude.values,ds.latitude.values, i,j)#i, xm[nearestX])
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
    conf = getConfigurationByID('.', 'model_preproc')
    conf_pre=getConfigurationByID('.','sat_preproc')

    years=f"{conf_pre.years[0]}_{conf_pre.years[-1]}" if len(conf_pre.years)>1  else conf_pre.years[0]

    outdir = conf.out_dir
    os.makedirs(outdir, exist_ok=True)

    sat_path=(conf.datasets.sat.path).format(year=years,sigma=conf_pre.processing.filters.zscore.sigma)
    sat=xr.open_dataset(sat_path)

    for dataset in conf.datasets.models:
        outname=os.path.join(outdir,'{ds}_{yr}'.format(ds=dataset,yr=years))
        print ('processing %s dataset')
        filledPath=(conf.datasets.models[dataset].path).format(experiment=dataset,year=years)
        print ('searching for ', filledPath)
        ds=xr.open_mfdataset(natsorted(glob(filledPath)),combine='by_coords')
        # get timeseries
        getSeries(ds,sat,conf,dataset,outname)

if __name__ == '__main__':
    main()
