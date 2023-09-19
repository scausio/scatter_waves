import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID,checkOutdir
import numpy as np
from natsort import natsorted
from datetime import  datetime


def myMFdataset(fileList,conf,dataset):
    out=[xr.open_dataset(f) for f in fileList]
    return xr.concat(out,dim='time')

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

    return ds[conf.datasets.models[dataset].lon], ds[conf.datasets.models[dataset].lat], ds[conf.datasets.models[dataset].time]

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

    #idx=[]
    series = []
    for i, j, k in sat_idxs:
        nearestX=np.argmin(np.abs(xm - i))
        nearestY=np.argmin(np.abs(ym - j))
        nearestT=np.argmin(np.abs(tm - k))

        dx=xm[nearestX]-i
        dy=ym[nearestY]-j
        dt=tm[nearestT]-k

        if (dx>float(conf.filters.max_distance_in_space)) or (dy>float(conf.filters.max_distance_in_space)) or (((dt / np.timedelta64(1, 'h'))>conf.filters.max_distance_in_time)):
            series.append(np.nan)
            print ('skip point')
        else:
            subset=model[conf.datasets.models[dataset].hs].isel(**{conf.datasets.models[dataset].lon:nearestX,conf.datasets.models[dataset].lat:nearestY,conf.datasets.models[dataset].time:nearestT})

            series.append(subset.compute().data)
            # idx.append([nearestX,nearestY,nearestT])


            print('time:', dt / np.timedelta64(1, 'h'))#k, tm[nearestT])
            print('x:', dx)#i, xm[nearestX])
            print('y:', dy)#)j, ym[nearestY])
    #series=np.clip(series,conf.filters.threshold.min,conf.filters.threshold.max)

    # idx=np.array(idx)
    # model[conf.datasets.models[dataset].hs].isel(
    #         **{conf.datasets.models[dataset].lon: idx[0], conf.datasets.models[dataset].lat: idx[1],
    #            conf.datasets.models[dataset].time: idx[2]})
    series=np.array(series)
    series[series <conf.filters.threshold.min]=np.nan
    series[series>conf.filters.threshold.max]=np.nan
    np.save('%s_series.npy'%outname,series)

def fromNEMOcoords(ds):
    navlat = ds['nav_lat'].values
    navlon = ds['nav_lon'].values
    ds['lat'] = np.unique(navlat.ravel())
    ds['lon'] = np.unique(navlon.ravel())

    ds = ds.rename({'x': 'lon', 'y': 'lat','time_counter':'time'}).set_coords(['lat', 'lon'])
    ds = ds.drop('nav_lat')
    ds = ds.drop('nav_lon')

    ds['lon'].attrs['standard_name'] = 'longitude'
    ds['lat'].attrs['standard_name'] = 'latitude'
    try:
        formatted_time = [datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S') for t in
                          ds.time.values]
    except:
        formatted_time = [datetime.strptime(str(t).split('.')[0], '%Y-%m-%dT%H:%M:%S') for t in
                          ds.time.values]
    ds['time'] = formatted_time
    return ds

def main():
    conf = getConfigurationByID('.', 'model_preproc')
    checkOutdir(conf.out_dir)
    years=f"{conf.years[0]}_{conf.years[-1]}" if len(conf.years)>1  else conf.years[0]
    outdir=conf.out_dir
    sat_path=(conf.datasets.sat.path).format(year=years,sigma=conf.processing.zscore.sigma)
    sat=xr.open_dataset(sat_path)
    for dataset in conf.datasets.models:
        outname=os.path.join(outdir,'{ds}_{yr}'.format(ds=dataset,yr=years))
        print ('processing %s dataset'%dataset)
        filledPath=(conf.datasets.models[dataset].path).format(experiment=dataset,year=years)
        conv_dir=os.path.join(conf.out_dir,'cf_compl')
        os.makedirs(conv_dir,exist_ok=True)
        for nc in natsorted(glob(filledPath)):
            outNCconv_name=os.path.join(conv_dir,os.path.basename(nc))
            #if not os.path.exists(outNCconv_name):
            print ('converting %s'%nc)
            ds=xr.open_dataset(nc)
            subset=fromNEMOcoords(ds)[conf.datasets.models[dataset].hs]
            subset.to_netcdf(outNCconv_name)

        print ('conversion completed')
        ds=myMFdataset(natsorted(glob(os.path.join(conv_dir,'*.nc'))),conf,dataset)
        print ('file open')

        # get timeseries
        getSeries(ds,sat,conf,dataset,outname)

if __name__ == '__main__':
    main()
