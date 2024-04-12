import xarray as xr
import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID,daysBetweenDates
import numpy as np
from natsort import natsorted
from scipy.spatial import KDTree


class Reader():
    def __init__(self,conf,model,dataset,sat_points):
        self.conf=conf
        self.model=model
        self.dataset=dataset
        self.sat_points=sat_points
        self.read()

    def nemo(self):
        pass
    def cf(self):
        hs=self.model[self.conf.datasets.models[self.dataset].hs].isel(time=0)
        land=np.isnan(hs.flatten())
        all_lon=self.model[self.conf.datasets.models[self.dataset].lon].values
        all_lat=self.model[self.conf.datasets.models[self.dataset].lat].values
        xx,yy=np.meshgrid(all_lon,all_lat)
        lon=xx.flatten()[~land]
        lat=yy.flatten()[~land]
        hs=hs[~land]
        self.run(lon, lat, hs)

    def unstruct(self):
        lon=self.model[self.conf.datasets.models[self.dataset].lon].values
        lat= self.model[self.conf.datasets.models[self.dataset].lat].values
        hs=self.model[self.conf.datasets.models[self.dataset].hs].values
        self.run(lon,lat,hs)

    def run(self,lon,lat,hs):
        print('building kdtree')

        tree = KDTree(np.array((lon,lat)).T)
        #
        print('getting nearest in space')

        dist, idxs = tree.query(self.sat_points, k=1)

        print('filtering in space')

        self.mask_dist = np.abs(dist) <= float(self.conf.filters.max_distance_in_space)

        self.model_hs = hs
        self.idxs=idxs
        print (self.model_hs.shape)
        return self

    def read(self):
        tp=self.conf.datasets.models[self.dataset].type
        av_tp=['u','unstr','unstruct','unstructured']
        if not tp.lower() in av_tp:
            exit(f'Please check model type. Your selection is: {tp}, available types are {av_tp}')
        elif tp.lower() in ['u','unstr','unstruct','unstructured']:
            print  ('you are using unstructured model')
            self.unstruct()
        elif tp.lower() in ['nemo','nemo','nemo','nemo']:
            print  ('you are using unstructured model')
            self.nemo()
        elif tp.lower() in ['cf','cf_compl','cf_compliant']:
            print  ('you are using unstructured model')
            self.cf()

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
    try:
        sat_points = np.array((sat_idxs[time_filt, 0], sat_idxs[time_filt, 1])).T
    except:
        return
    print(np.array(sat_idxs).shape)

    data=Reader(conf,model,dataset,sat_points)
    print (data.model_hs.shape)
    print('slicing in time')
    print (time_idxs.shape)
    print (time_filt.shape)
    print (data.mask_dist.shape)
    print(sat['model_hs'].values.shape)
    model_hs = data.model_hs[time_idxs[time_filt][data.mask_dist],data.idxs[data.mask_dist]]
    if model_hs.shape==0:
        return
    print (model_hs.shape)
    print('slicing sat')
    sat=sat.isel(obs=time_filt).isel(obs=data.mask_dist).sel(model=[dataset])
    print('replacing')
    print (sat)
    print (model_hs.shape,sat['model_hs'].values.shape)
    sat['model_hs'].values=np.array([model_hs]).T
    print (1)
    sat['hs'].attrs['satellite_file'] = satname
    print (2)
    sat.to_netcdf('%s' % outname)
    print (3)
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
     
    outname_all=os.path.join(outdir,'{date}_series.nc'.format(date=date))
    if not os.path.exists(outname_all):
        buffer_all=[]
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
                        buffer.append(daily_ds)
                        try:
                            ds = preprocesser(xr.open_dataset(filledPath))
                            daily_ds = getSeries(ds, sat, conf_model,conf_sat, dataset, os.path.basename(sat_path), outname_day)
                            buffer.append(daily_ds)
                        except:
                            pass
                    else:
                        buffer.append(xr.open_dataset(outname_day))
                ds_model_out=xr.concat(buffer,dim='obs')
                buffer_all.append(ds_model_out)
                ds_model_out.to_netcdf(outname)
            else:
                buffer_all.append(xr.open_dataset(outname))

        ds_out=xr.merge(buffer_all)
        ds_out['model']=[ds for ds in conf_model.datasets.models]
        ds_out.to_netcdf(outname_all)

