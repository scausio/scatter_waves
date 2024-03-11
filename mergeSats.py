import xarray as xr
import numpy as np
import os
from glob import glob
import munch
import yaml
from natsort import natsorted
from utils import getConfigurationByID


# def myMFdataset(fileList,conf,conf_model):
#     #out=[xr.open_dataset(f).VAVH for f in fileList]
#     #print (out)
#     obs=[]
#     time=[]
#     lon=[]
#     lat=[]
#     hs=[]
#     n=0
#     for f in fileList:
#         print (f)
#         fo=xr.open_dataset(f)
#         print (fo)
#         for i,t in enumerate(fo[conf.sat_specifics.time]):
#             obs.append(n)
#             time.append(t.data)
#             lon.append(fo[conf.sat_specifics.lon].isel(time=i).data)
#             lat.append(fo[conf.sat_specifics.lat].isel(time=i).data)
#             hs.append(fo[conf.sat_specifics.hs].isel(time=i).data)
#             n+=1
#     base = xr.Dataset(data_vars=dict(obs=(['obs'],obs),
#                                       hs=(['obs'],hs),
#                                       time=(['obs'],time),
#                                       longitude=(['obs'],lon),
#                                       latitude=(['obs'],lat)))
#
#     base.coords['model'] = np.array(list(conf_model.datasets.models.keys()), dtype=str)
#     model_variable = np.zeros_like(base['hs'], shape=tuple(base.sizes.values()))
#     base['model_hs'] = xr.DataArray(model_variable, dims=base.sizes.keys())
#     return base


def myMFdataset(fileList,conf,conf_model):

    mrgd=xr.open_dataset(fileList[0])
    obs=np.arange(len(mrgd.time.values))
    mrgd['obs']=('time',obs)
    mrgd= mrgd.swap_dims({'time':'obs'}).reset_index('obs')
    print (len(mrgd.obs.values))
    print (mrgd)
    for f in fileList[1:]:
        print (f )
        ds=xr.open_dataset(f)
        obs=np.arange(len(ds.time.values))+(obs[-1]+1)
        ds['obs']=('time',obs)
        ds = ds.swap_dims({'time':'obs'}).reset_index('obs')
        mrgd=xr.concat([mrgd, ds],'obs')
        print (len(mrgd.obs.values))
        print (ds)

    mrgd=mrgd.rename({conf.sat_specifics.lon:'longitude',conf.sat_specifics.lat:'latitude',
                     conf.sat_specifics.hs:'hs',conf.sat_specifics.time:'time'})
    mrgd.coords['model'] = np.array(list(conf_model.datasets.models.keys()), dtype=str)
    model_variable = np.zeros_like(mrgd['hs'], shape=tuple(mrgd.sizes.values()))
    mrgd['model_hs'] = xr.DataArray(model_variable, dims=mrgd.sizes.keys())
    return mrgd


conf=getConfigurationByID('.','sat_preproc')
conf_model=getConfigurationByID('.','model_preproc')
base=conf.paths.output
fname_tmpl='*_landMasked_qcheck_zscore{sigma}.nc'.format(sigma=conf.processing.filters.zscore.sigma)

ff=natsorted(glob(os.path.join(base,fname_tmpl)))

mrgd=myMFdataset(ff,conf,conf_model)
print (mrgd)
years=f"{conf.years[0]}_{conf.years[-1]}" if len(conf.years)>1  else conf.years[0]
mrgd.to_netcdf(os.path.join(base,'{yy}_CMEMS_SAT_{tmpl}'.format(yy=years,tmpl=fname_tmpl[2:])))
