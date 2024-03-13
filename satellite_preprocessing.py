from shapely.geometry import Point,Polygon, shape
import fiona
import xarray as xr
import numpy as np
import os
import glob
from utils import getConfigurationByID,daysBetweenDates
import time
from scipy import stats
from glob import glob
from natsort import natsorted
import geopandas as gpd
import shapefile as shp

start_time = time.time()

def maskOutliers(ds,filters):
    satVar = ds.values
    msk=~np.isnan(satVar)
    noNanSatvar=satVar[msk]
    print('nan before masking:', np.cumsum(np.isnan(satVar)))
    z = np.abs(stats.zscore(noNanSatvar))
    noNanSatvar[z >filters.zscore.sigma]=np.nan
    satVar[msk]=noNanSatvar
    # print('nan after masking:', np.cumsum(np.isnan(satVar)))
    # ds.values=satVar
    return  ~np.isnan(satVar)

def pointInPoly_old(shp_path, points):
    shp = fiona.open(shp_path, 'r')
    points = [Point(i, j) for i, j in points]
    inPoly=[point.within(shape(shp[0]['geometry'])) for point in points]
    return np.array(inPoly)

def pointInPoly(shp_path, points):
    shp = gpd.read_file(shp_path)
    mask=points[:,0]*0
    data_gdf = gpd.GeoDataFrame(mask, geometry=gpd.points_from_xy(points[:,0],points[:,1]))
    idx = gpd.clip(data_gdf, shp).index
    mask=mask.astype(bool)
    mask[idx]=True
    return np.array(mask)


def findTrackIdInBox(conf,folder, fs):
    minLon = conf.processing.boundingBox.xmin
    maxLon = conf.processing.boundingBox.xmax
    minLat = conf.processing.boundingBox.ymin
    maxLat = conf.processing.boundingBox.ymax
    latName=conf.sat_specifics.lat
    lonName=conf.sat_specifics.lon
    inBox = []
    for i, f in enumerate(fs):
        nc = xr.open_dataset(os.path.join(folder, f))
        if (np.nanmax(np.abs(nc[latName].values)) >= np.abs(minLat)) & (
                np.nanmin(np.abs(nc[latName].values)) <= np.abs(maxLat)) \
                & (np.nanmin(np.abs(nc[lonName].values)) <= np.abs(maxLon)) & (
                np.nanmax(np.abs(nc[lonName].values)) >= np.abs(minLon)):
            inBox.append(i)
    return np.array(inBox)


def cutTrackOnBox(conf,nc):
    minLon = conf.processing.boundingBox.xmin
    maxLon = conf.processing.boundingBox.xmax
    minLat = conf.processing.boundingBox.ymin
    maxLat = conf.processing.boundingBox.ymax
    latName=conf.sat_specifics.lat
    lonName=conf.sat_specifics.lon
    #print ('subsetting... %s'%nc)
    nc=xr.open_dataset(nc)
    lon=nc[lonName].values
    lon[lon>180]-=360
    nc[lonName].values=lon
    
    msk = np.where((nc[latName].data > minLat) & (nc[latName].data < maxLat) & (nc[lonName].data > minLon) & (nc[lonName].data < maxLon))[0]
    if len(msk)==0:
        return False
    else:
        return nc.isel(**{conf.sat_specifics.time:msk})


def getTracksInBox(conf, files):
    """
    :param conf: configuration file
    :param files: list of netcdf to be processed
    :return: netcdf file with a sat track within the selected bounding box
    """

    n=len(files)
    if n==0:
        print ('no files found')
        return

    print ('%s files to go'% n)
    for i,f in enumerate(files):
        nc = cutTrackOnBox(conf, f)
        if nc:
            break
        else:
            pass
        n -= 1

    for f in files[i+1:]:
        nc_ = cutTrackOnBox(conf,  f)
        if nc_:
            nc = xr.concat([nc, nc_], dim=conf.sat_specifics.time)
        else:
            pass

    return nc

def saveNc(ncs, outname):
    ncs.to_netcdf( outname)




def getLand(ds, maskValue):
    return ds.data!=maskValue

def getFilenames(conf,day,sat_name):
    return os.path.join(conf.paths.sat,conf.filenames.sat.template).format(satName=sat_name,
                                              year=day[:4],
                                              satType=conf.sat_specifics.type,
                                              month=day[4:6],day=day)

class Sat_processer:
    def __init__(self,conf_path,start_date,end_date):
        self.conf = getConfigurationByID(conf_path, 'sat_preproc')
        self.conf_model = getConfigurationByID(conf_path, 'model_preproc')
        self.days=daysBetweenDates(start_date, end_date)
        print (self)
        os.makedirs(self.conf.paths.out_dir, exist_ok=True)
        self.start_date=start_date
        self.end_date=end_date
        #self.years=np.arange(self.conf.years[0],self.conf.years[-1]+1,1)

    def trackInArea(self,sat_name,day):

        outname= "%s.nc" % os.path.join(self.conf.paths.out_dir,
                                           self.conf.filenames.output.format(sat_name=sat_name, day=day))
        print (outname)
        if os.path.exists(outname):
            return xr.open_dataset(outname)
        else:
            name_tmpl = getFilenames(self.conf, day, sat_name)
            fs = glob(name_tmpl)
            print(len(fs), 'files found')
            # get nc of tracks only in the box
            ds = getTracksInBox(self.conf, fs)
            if ds:
                saveNc(ds, outname)

            print("--- %s seconds ---" % (time.time() - start_time))
            return ds


    def masking(self,ds,sat_name,day):
        self.filters = self.conf.processing.filters
        lat=self.conf.sat_specifics.lat
        lon=self.conf.sat_specifics.lon
        time = self.conf.sat_specifics.time
        self.hs=self.conf.sat_specifics.hs
        # apply land mask
        outname= "%s_landMasked_qcheck.nc" % os.path.join(self.conf.paths.out_dir,
                                                                       self.conf.filenames.output.format(sat_name=sat_name,
                                                                                                    day=day))
        if not os.path.exists(outname):
            print('quality check')
            if self.filters.quality_check.variable_name:
                qcMsk = np.where(ds[self.filters.quality_check.variable_name].data != self.filters.quality_check.value)
                ds[self.hs][qcMsk] = np.nan

            if self.filters.land_masking.shapefile:
                print('mask from shp')
                landPoints = np.argwhere(
                    ~np.array(pointInPoly(self.filters.land_masking.shapefile, np.array([ds[self.hs][lon].values, ds[self.hs][lat].values]).T)))
                ds[self.hs].values[landPoints] = np.nan

            if self.filters.land_masking.variable_name:
                print('mask from variable')
                landPoints = getLand(ds[self.filters.land_masking.variable_name], self.filters.land_masking.value)
                ds[self.hs].data[landPoints] = np.nan

            # ds[self.hs].data[ds[self.hs].data < self.filters.threshold.min] = np.nan
            # ds[self.hs].data[ds[self.hs].data > self.filters.threshold.max] = np.nan
            # plt.scatter(ds[hs][lon], ds[hs][lat],c=ds[hs].data)
            # plt.show()
            ds=ds[[self.hs,time,lon,lat]]
            saveNc(ds, outname)
        return xr.open_dataset(outname)

    def ZScore(self,ds,outname):
        outname= f"{outname}"
        print (outname)
        if not os.path.exists(outname):
            # apply zscore
            print('masking outliers')
            idx= maskOutliers(ds['hs'], self.filters)
            #saveNc(ds, outname)
            #ds_msk = xr.open_dataset(outname)
            #m = np.argwhere(~np.isnan(ds_msk['hs'].data))[:, 0]
            ds = ds.isel(obs=idx)
            #print(m.shape)
            # ds=ds.isel(**{conf.sat_specifics.time:m})
            #os.remove(outname)
            # plt.scatter(ds[hs][lon], ds[hs][lat], c=m, s=1)
            # plt.show()
            print (ds)
            saveNc(ds, outname)

    def merge_sats(self):
        base = self.conf.paths.out_dir
        fname_tmpl = f'*_landMasked_qcheck.nc'
        date = f"{self.start_date}_{self.end_date}"
        if not os.path.exists(os.path.join(base, '{yy}_{tmpl}_ALLSAT.nc'.format(yy=date, tmpl=fname_tmpl[2:-3]))):
            fileList = natsorted(glob(os.path.join(base, fname_tmpl)))

            mrgd = xr.open_dataset(fileList[0])
            obs = np.arange(len(mrgd.time.values))
            mrgd['obs'] = ('time', obs)
            mrgd = mrgd.swap_dims({'time': 'obs'}).reset_index('obs')
            print('first file len:',len(mrgd.obs.values))

            for f in fileList[1:]:
                print(f)
                ds = xr.open_dataset(f)
                obs = np.arange(len(ds.time.values)) + (obs[-1] + 1)
                ds['obs'] = ('time', obs)
                ds = ds.swap_dims({'time': 'obs'}).reset_index('obs')
                print('ds file len:',len(ds.obs.values))

                mrgd = xr.concat([mrgd, ds], 'obs').reset_index('obs')

                print('mrg file len:',len(mrgd.obs.values))


            mrgd = mrgd.rename({self.conf.sat_specifics.lon: 'longitude', self.conf.sat_specifics.lat: 'latitude',
                                self.conf.sat_specifics.hs: 'hs', self.conf.sat_specifics.time: 'time'})
            mrgd.coords['model'] = np.array(list(self.conf_model.datasets.models.keys()), dtype=str)
            model_variable = np.zeros_like(mrgd['hs'], shape=tuple(mrgd.sizes.values()))
            mrgd['model_hs'] = xr.DataArray(model_variable, dims=mrgd.sizes.keys())
            mrgd['time'].values = mrgd['time'].dt.round(freq='H')
            mrgd.to_netcdf(os.path.join(base, '{yy}_{tmpl}_ALLSAT.nc'.format(yy=date, tmpl=fname_tmpl[2:-3])))
        else:
            mrgd=xr.open_dataset(os.path.join(base, '{yy}_{tmpl}_ALLSAT.nc'.format(yy=date, tmpl=fname_tmpl[2:-3])))
        self.ZScore(mrgd, os.path.join(base, '{yy}_{tmpl}_zscore{sigma}_ALLSAT.nc'.format(
            yy=date, tmpl=(fname_tmpl[2:-3]),sigma=self.conf.processing.filters.zscore.sigma)))

    def run(self):
        date = f"{self.start_date}_{self.end_date}"
        print (os.path.join(self.conf.paths.out_dir, '{date}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc'.format(
            date=date,sigma=self.conf.processing.filters.zscore.sigma)))

        if not os.path.exists(os.path.join(self.conf.paths.out_dir, '{date}_landMasked_qcheck_zscore{sigma}_ALLSAT.nc'.format(
            date=date,sigma=self.conf.processing.filters.zscore.sigma))):
            for sat_name in self.conf.sat_names:
                print(f"Processing {sat_name}")
                for day in self.days:
                    print(f"Processing {day}")
                    ds=self.trackInArea(sat_name, day)
                    if ds:
                        ds=self.masking(ds, sat_name, day)
            self.merge_sats()
