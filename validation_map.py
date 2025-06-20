import matplotlib
matplotlib.use('Agg')
from stats import  metrics
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import getConfigurationByID,ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def coords2bins(ds_all,x,y,step):
    y_min, y_max = np.nanmin(ds_all.latitude.values)-step, np.nanmax(ds_all.latitude.values)+step
    x_min, x_max = np.nanmin(ds_all.longitude.values)-step, np.nanmax(ds_all.longitude.values)+step
    lon_bins = np.arange(x[np.argmin(np.abs(x - x_min))], x[np.argmin(np.abs(x - x_max))] + step, step)
    lat_bins = np.arange(y[np.argmin(np.abs(y - y_min))], y[np.argmin(np.abs(y - y_max))] + step, step)
    return lon_bins,lat_bins

def targetGrid(step):
    x = np.arange(-180-(step/2), 180 + step, step)
    y = np.arange(-90-(step/2), 90 + step, step)
    return x,y

def get2Dbins(data,step):
    print(f'target grid definition at {step} of resolution')
    x,y=targetGrid(step)
    data = data.sortby('latitude')
    print('2-dimensional binning')
    lon_bins,lat_bins=coords2bins(data,x,y, step)
    grouped_lon = data.groupby_bins('longitude', lon_bins)
    print('lon binning')
    lon_=[i.mid for i in grouped_lon.groups.keys()]
    buffer=[]
    print ('lat binning')

    for i,lon_group in enumerate(grouped_lon._iter_grouped()):
        print (len(list(grouped_lon._iter_grouped()))-i)
        lat_group=lon_group.groupby_bins('latitude', lat_bins)
        lat_ = [i.mid for i in lat_group.groups.keys()]
        results=lat_group.apply(metrics)
        results = results.rename(latitude_bins='latitude')
        results=results.assign_coords({"latitude": [i.mid for i in results.latitude.values]})
        buffer.append(results)
    print('reordering lon')
    out=xr.concat([buffer[i] for i in np.argsort(lon_)], 'longitude')
    print('redefining lon')
    out=out.assign_coords({"longitude": np.array(lon_)[np.argsort(lon_)]})
    return out.transpose()

def plotMap(ds, variable,coast_resolution, outfile):
    print ('Plotting map')
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    m = Basemap(llcrnrlon=ds.longitude.min()-0.25, llcrnrlat=ds.latitude.min()-0.25,
                urcrnrlat=ds.latitude.max()+0.25, urcrnrlon=ds.longitude.max()+0.25, resolution=coast_resolution)
    m.drawcoastlines()
    m.fillcontinents('Whitesmoke')
    tick_spacing=ticker(ds.latitude.min(),ds.latitude.max())
    meridians = np.arange(-180, 180+tick_spacing, tick_spacing)
    parallels = np.arange(-90,90+tick_spacing, tick_spacing)
    m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
    m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
    if variable=='bias':
        plt.title('Bias [m]', loc='left')
        vmin=-1
        vmax=1
        cmap='seismic'
        var=ds[variable]
    elif variable == 'rmse':
        plt.title('RMSE [m]', loc='left')
        vmin=0
        vmax=1
        cmap = 'jet'
        var=ds[variable]
    elif variable == 'nrmse':
        plt.title(f'NRMSE [%]: {str(np.nanmean(ds[variable]*100))[:5]}', loc='left')
        vmin=0
        vmax=60
        cmap = 'jet'
        var=ds[variable]*100
    elif variable=='nbias':
        plt.title(f'NBias [%]: {str(np.nanmean(ds[variable]*100))[:5]}', loc='left')
        vmin=-60
        vmax=60
        cmap='seismic'
        var=ds[variable]*100

    im= plt.imshow(var, origin='lower', cmap=cmap, vmin=vmin,
                         vmax=vmax,
                         extent=[ds.longitude.min(), ds.longitude.max(), ds.latitude.min(), ds.latitude.max()])
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(outfile)
    plt.close()




# plt.show()

def main(conf_path, start_date, end_date):

    conf = getConfigurationByID(conf_path, 'plot')
    outdir = conf.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir,'plots'), exist_ok=True)
    date = f"{start_date}_{end_date}"

    for dataset in conf.experiments:
        print(dataset)
        ds_all = xr.open_dataset((conf.experiments[dataset].series).format(out_dir=outdir,date=date, experiment=dataset)).sel(
            model=dataset)

        sat_hs = ds_all.hs

        model_hs = ds_all.model_hs
        if conf.filters.unbias in ['True','T','TRUE','t']:
            sat_hs-=np.nanmean(sat_hs)
            sat_hs+=np.nanmean(model_hs)

        print(np.sum(np.isnan(sat_hs)))

        ds = get2Dbins(ds_all, conf.binning_resolution)
        print('saving output')

        for variable in ['nbias','nrmse']:
            outName = os.path.join(outdir,'plots', f'{variable}_{dataset}_{date}.jpeg')
            plotMap(ds, variable, conf.coast_resolution, outName)

