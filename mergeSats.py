import xarray as xr
import numpy as np
import os
from glob import glob
import munch
import yaml

def getConfigurationByID(path,confId):
    globalConf = yaml.load(open(os.path.join(path,"config.yaml")))
    return munch.Munch.fromDict(globalConf[confId])


def myMFdataset(fileList,conf):
    #out=[xr.open_dataset(f).VAVH for f in fileList]
    #print (out)
    obs=[]
    time=[]
    lon=[]
    lat=[]
    hs=[]
    n=0
    for f in fileList:
        print (f)
        fo=xr.open_dataset(f)
        print (fo)
        for i,t in enumerate(fo[conf.sat_specifics.time]):
            obs.append(n)
            time.append(t.data)
            lon.append(fo[conf.sat_specifics.lon].isel(time=i).data)
            lat.append(fo[conf.sat_specifics.lat].isel(time=i).data)
            hs.append(fo[conf.sat_specifics.hs].isel(time=i).data)
            n+=1
    result = xr.Dataset(data_vars=dict(obs=(['obs'],obs),
                                      hs=(['obs'],hs),
                                      time=(['obs'],time),
                                      longitude=(['obs'],lon),
                                      latitude=(['obs'],lat))) 
    #result['obs']=['obs',obs]
    #result['time']=['obs',time]
    #result['longitude']=['obs',lon]
    #result['latitude']=['obs',lat]
    return result

conf=getConfigurationByID('.','sat_preproc')
base=conf.paths.output
fname_tmpl='*_threshold_landMasked_qcheck_zscore3.nc'

ff=glob(os.path.join(base,fname_tmpl))
#xr.open_mfdataset(ff, combine='by_coords',concat_dim='obs')
mrgd=myMFdataset(ff,conf)
print (mrgd)
years=f"{conf.years[0]}_{conf.years[-1]}" if len(conf.years)>1  else conf.years[0]
mrgd.to_netcdf(os.path.join(base,'{yy}_CMEMS_SAT_{tmpl}'.format(yy=years,tmpl=fname_tmpl[2:])))
