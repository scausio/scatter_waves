import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import getConfigurationByID,ticker
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def varConversion(variable, variable_name):
    if variable_name in ['bias','rmse']:
        return variable.values
    elif variable_name in ['nbias','nrmse']:
        return variable.values*100

def plotTracks(ds, variable,coast_resolution, outfile,daily_track=False):
    x=ds.longitude.values
    y=ds.latitude.values
    fig = plt.figure()
    ax=plt.gca()
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
    ds['day'] = ('obs', ds.time.dt.floor('d').values)

    if variable=='bias':
        plt.title('Bias [m]', loc='left')
        vmin=-1
        vmax=1
        cmap='seismic'
    elif variable == 'rmse':
        plt.title('RMSE [m]', loc='left')
        vmin=0
        vmax=1
        cmap = 'jet'
    elif variable == 'nrmse':
        plt.title(f'NRMSE [%]: {str(np.nanmean(ds[variable]*100))[:5]}', loc='left')
        vmin=0
        vmax=60
        cmap = 'jet'
    elif variable=='nbias':
        plt.title(f'NBias [%]: {str(np.nanmean(ds[variable]*100))[:5]}', loc='left')
        vmin=-60
        vmax=60
        cmap='seismic'

    if daily_track:
        for i,day in enumerate(np.unique(ds['day'])):
            ds_daily=ds.isel(obs=ds['day']==day)
            var=varConversion(ds_daily[variable],variable)
            im=plt.scatter(ds_daily.longitude.values,ds_daily.latitude.values,c=var,cmap=cmap,vmin=vmin, vmax=vmax)
            if i==100000:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            title=str(day)[:10]
            outfile_daily = f'{outfile[:-23]}_{title}.jpeg'
            plt.title(title)
            plt.savefig(outfile_daily)
            del(im)
    else:
        im = plt.scatter(x, y, c=varConversion(ds[variable],variable), cmap=cmap, vmin=vmin, vmax=vmax)
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
        sat_hs = sat_hs.where(
          (ds_all.hs.values <= float(conf.filters.max)) & (ds_all.hs.values >= float(conf.filters.min)))
        model_hs = model_hs.where(
          (ds_all.model_hs.values <= float(conf.filters.max)) & (ds_all.model_hs.values >= float(conf.filters.min)))
        ds_all.hs.values=sat_hs
        ds_all.model_hs.values = model_hs
        ds_all['bias']=model_hs-sat_hs
        ds_all['nbias'] = (model_hs - sat_hs)/sat_hs
        ds_all['rmse']=np.sqrt((model_hs**2)-(sat_hs**2))
        ds_all['nrmse'] =  ds_all['rmse']/sat_hs

        for variable in ['nbias','nrmse','bias','rmse']:
            outName = os.path.join(outdir,'plots', f'tracks_{variable}_{dataset}_{date}.jpeg')
            plotTracks(ds_all, variable,conf.coast_resolution, outName, daily_track=False)

