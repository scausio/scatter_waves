from shapely.geometry import Point,Polygon, shape
import fiona
import xarray as xr
import numpy as np
import os
import glob
from utils import getConfigurationByID,checkOutdir
import time
import matplotlib.pyplot as plt
from scipy import stats

start_time = time.time()


def maskOutliers(ds,filters):
    satVar = ds.data
    msk=~np.isnan(satVar)
    noNanSatvar=satVar[msk]
    print('nan before masking:', np.cumsum(np.isnan(satVar)))
    z = np.abs(stats.zscore(noNanSatvar))
    noNanSatvar[z >=filters.zscore.sigma]=np.nan
    satVar[msk]=noNanSatvar
    print('nan after masking:', np.cumsum(np.isnan(satVar)))
    ds.data=satVar
    return ds

def pointInPoly(shp_path, points):
    shp = fiona.open(shp_path, 'r')
    print(len(shp))

    #poly = Polygon(shp[0]['geometry']['coordinates'][0])
    points = [Point(i, j) for i, j in points]
    inPoly=[point.within(shape(shp[0]['geometry'])) for point in points]
    #inPoly = [poly.contains(p) for p in points]
    return np.array(inPoly)

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
    print ('subsetting... %s'%nc)
    nc=xr.open_dataset(nc)
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

def filterTracksOnBox(conf,base, fs):
    """

    :param conf:
    :param base:
    :param fs:
    :return: filenames
    """
    idOnBox = findTrackIdInBox(conf,base, fs)
    boxTracks = np.array(fs)[idOnBox]
    print ('Cycles in box:%s' % set([i.split('_')[6] for i in boxTracks]))
    print ('files filtered')
    return boxTracks

def getLand(ds, maskValue):
    return ds.data!=maskValue

def getFilenames(conf,year,sat_name):
    #return conf.filenames.sat.template.format(cicle=conf.filenames.sat.cicle,
    #                                          passage=conf.filenames.sat.passage,
    #                                          year=conf.year)
    #CFOSAT/L3/sec/2019/07/
    return os.path.join(conf.paths.sat,conf.filenames.sat.template).format(satName=sat_name,
                                              year=year,
                                              satType=conf.sat_specifics.type,
                                              month="*")

def plotTracks():
    pass

def main():
    """
    The tool takes all file in the folder specified in conf.yaml sat path , checks if the sat swap falls into the bounding box and merge them.
    You can additionally subsetting the dataset according year, cycle or satellite pass
    The subsetting is saved as netcdf file
    :return:
    """
    conf=getConfigurationByID('.','sat_preproc')
    checkOutdir(conf.paths.output)

    lat=conf.sat_specifics.lat
    lon=conf.sat_specifics.lon
    hs=conf.sat_specifics.hs

    years=np.arange(conf.years[0],conf.years[-1]+1,1)

    filters = conf.processing.filters
    for sat_name in conf.sat_names:
        print (sat_name)
        for year in years:
            print (year)
            outname_1="%s.nc"%os.path.join(conf.paths.output,conf.filenames.output.format(sat_name=sat_name,year=year))

            if not os.path.exists(outname_1):
                name_tmpl=getFilenames(conf,year,sat_name)
                print (name_tmpl)

                fs = glob.glob(name_tmpl)
                print (len(fs), 'files found')
                # get nc of tracks only in the box
                ds=getTracksInBox(conf, fs)
                if not ds:
                    continue
                print("--- %s seconds ---" % (time.time() - start_time))
                saveNc(ds, outname_1)
            else:
                ds = xr.open_dataset(outname_1)

            # apply thresholds and land mask
            outname_2 ="%s_threshold_landMasked_qcheck.nc" % os.path.join(conf.paths.output,
                                   conf.filenames.output.format(sat_name=sat_name, year=year))

            if not os.path.exists(outname_2):
                print ('quality check')
                if filters.quality_check.variable_name:
                    qcMsk = np.where(ds[filters.quality_check.variable_name].data != filters.quality_check.value)
                    ds[hs][qcMsk] = np.nan

                if filters.land_masking.shapefile:
                    print ('mask from shp')
                    landPoints = np.argwhere(~np.array(pointInPoly(filters.land_masking.shapefile, zip(ds[hs][lon], ds[hs][lat]))))
                    ds[hs].data[landPoints]= np.nan

                if filters.land_masking.variable_name:
                    print ('mask from variable')
                    landPoints = getLand(ds[filters.land_masking.variable_name], filters.land_masking.value)
                    ds[hs].data[landPoints]= np.nan

                ds[hs].data[ds[hs].data<filters.threshold.min]=np.nan
                ds[hs].data[ds[hs].data > filters.threshold.max] = np.nan
                #plt.scatter(ds[hs][lon], ds[hs][lat],c=ds[hs].data)
                #plt.show()
                saveNc(ds, outname_2)

            ds = xr.open_dataset(outname_2)
            outname_3 = "%s_threshold_landMasked_qcheck_zscore%s.nc" % (os.path.join(conf.paths.output,conf.filenames.output.format(sat_name=sat_name, year=year)),filters.zscore.sigma)
            if not os.path.exists(outname_3):
                # apply zscore
                print ('masking outliers')
                ds=maskOutliers(ds[hs],filters)
                saveNc(ds, outname_3)
                ds = xr.open_dataset(outname_3)
                print (ds)
                m = np.argwhere(~np.isnan(ds[hs].data))[:,0]
                print (m.shape)
                #ds=ds.isel(**{conf.sat_specifics.time:m})
                os.remove(outname_3)
                ds = ds.isel(time=m)
                # plt.scatter(ds[hs][lon], ds[hs][lat], c=m, s=1)
                # plt.show()
                saveNc(ds, outname_3)
                #plotTracks()

if __name__ == '__main__':
    main()

