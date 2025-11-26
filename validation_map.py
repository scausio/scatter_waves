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
import pandas as pd
from scipy import stats as scipy_stats

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

def get2Dbins_fast(data, step):
    """Optimazing version using pandas groupby instead of xarray groupby"""
    print(f'target grid definition at {step} of resolution')
    x, y = targetGrid(step)
    print('2-dimensional binning (optimized)')
    lon_bins, lat_bins = coords2bins(data, x, y, step)
    
    # Converting for DataFrame - more efficient for groupby
    print('Converting to DataFrame...')
    df = pd.DataFrame({
        'lon': data.longitude.values,
        'lat': data.latitude.values,
        'hs': data.hs.values,
        'model_hs': data.model_hs.values,
        'time': data.time.values
    })
    
    # Creating bins using pd.cut - vectorized and fast
    print('Creating bins...')
    df['lon_bin'] = pd.cut(df['lon'], bins=lon_bins, labels=False, include_lowest=True)
    df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins, labels=False, include_lowest=True)
    
    # Removing NaNs in bins
    df = df.dropna(subset=['lon_bin', 'lat_bin'])
    
    # Grouping and calculating metrics using vectorized operation
    print('Computing metrics (vectorized)...')
    grouped = df.groupby(['lon_bin', 'lat_bin'])
    
    results_dict = {}
    results_dict['bias'] = grouped.apply(lambda x: (x['model_hs'] - x['hs']).mean()).values
    results_dict['nbias'] = grouped.apply(lambda x: (x['model_hs'] - x['hs']).sum() / x['hs'].sum() if x['hs'].sum() > 0 else np.nan).values
    results_dict['rmse'] = grouped.apply(lambda x: np.sqrt(((x['model_hs'] - x['hs'])**2).mean())).values
    results_dict['nrmse'] = grouped.apply(lambda x: np.sqrt(((x['model_hs'] - x['hs'])**2).sum() / (x['hs']**2).sum()) if (x['hs']**2).sum() > 0 else np.nan).values
    results_dict['nobs'] = grouped.size().values
    results_dict['model_hs'] = grouped['model_hs'].mean().values
    results_dict['sat_hs'] = grouped['hs'].mean().values
    
    # Getting bin coordinates
    bin_coords = np.array(list(grouped.groups.keys()))
    lon_indices = bin_coords[:, 0].astype(int)
    lat_indices = bin_coords[:, 1].astype(int)
    
    # Calculating bin centers
    lon_centers = np.array([(lon_bins[i] + lon_bins[i+1])/2 for i in lon_indices])
    lat_centers = np.array([(lat_bins[i] + lat_bins[i+1])/2 for i in lat_indices])
    
    # Creating 2D grid
    print('Creating output grid...')
    n_lon = len(lon_bins) - 1
    n_lat = len(lat_bins) - 1
    
    # Initializing 2D arrays with NaN
    output_arrays = {key: np.full((n_lat, n_lon), np.nan) for key in results_dict.keys()}
    lon_grid = np.full((n_lat, n_lon), np.nan)
    lat_grid = np.full((n_lat, n_lon), np.nan)
    
    # Filling values
    for idx, (lon_idx, lat_idx) in enumerate(zip(lon_indices, lat_indices)):
        for key in results_dict.keys():
            output_arrays[key][lat_idx, lon_idx] = results_dict[key][idx]
        lon_grid[lat_idx, lon_idx] = lon_centers[idx]
        lat_grid[lat_idx, lon_idx] = lat_centers[idx]
    
    # Creating xarray Dataset at the end...
    print('Creating xarray Dataset...')
    ds = xr.Dataset(
        data_vars={key: (['latitude', 'longitude'], output_arrays[key]) for key in results_dict.keys()},
        coords={
            'longitude': (['latitude', 'longitude'], lon_grid),
            'latitude': (['latitude', 'longitude'], lat_grid)
        }
    )
    
    return ds

def get2Dbins(data, step):
    """Wrapper to keep compatibility with previous function"""
    return get2Dbins_fast(data, step)

def plotMap(ds, variable,coast_resolution, outfile):
    print ('Plotting map')
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    m = Basemap(llcrnrlon=ds.longitude.min()-0.25, llcrnrlat=ds.latitude.min()-0.25,
                urcrnrlat=ds.latitude.max()+0.25, urcrnrlon=ds.longitude.max()+0.25, resolution=coast_resolution)
    m.drawcoastlines()
    m.fillcontinents('Whitesmoke')
    tick_spacing=ticker(ds.latitude.min(),ds.latitude.max())
    meridians = np.arange(-180, 180+tick_spacing*2, tick_spacing*2)
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


def main(conf_path, start_date, end_date):

    conf = getConfigurationByID(conf_path, 'plot')
    outdir = conf.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir,'plots'), exist_ok=True)
    date = f"{start_date}_{end_date}"

    for dataset in conf.experiments:
        print(dataset)
        series_path = (conf.experiments[dataset].series).format(out_dir=outdir,date=date, experiment=dataset)
        if not os.path.exists(series_path):
            print(f"Series file not found for '{dataset}': {series_path}. Skipping map plot.")
            continue
        
        # Using chunks for lazy loading - avoids loading everything into memory
        print('Loading data with chunking...')
        ds_all = xr.open_dataset(series_path, chunks={'obs': 100000})
        
        if 'model' not in ds_all or dataset not in ds_all.model.values:
            print(f"Dataset '{dataset}' not present in 'model' coordinate. Available: {ds_all.model.values if 'model' in ds_all else 'None'}")
            continue
        if 'hs' not in ds_all or 'model_hs' not in ds_all:
            print(f"Missing required variables ('hs','model_hs') in series file for '{dataset}'. Variables: {list(ds_all.data_vars)}")
            continue
        ds_all = ds_all.sel(model=dataset)

        sat_hs = ds_all.hs

        model_hs = ds_all.model_hs
        if conf.filters.unbias in ['True','T','TRUE','t']:
            sat_hs-=np.nanmean(sat_hs)
            sat_hs+=np.nanmean(model_hs)

        print(np.sum(np.isnan(sat_hs)))
        
        # Load into memory only the necessary data
        print('Loading required data into memory...')
        ds_all = ds_all.compute()

        try:
            ds = get2Dbins(ds_all, conf.binning_resolution)
        except Exception as e:
            print(f"2D binning failed for '{dataset}': {e}")
            continue
        print('saving output')

        for variable in ['nbias','nrmse']:
            outName = os.path.join(outdir,'plots', f'{variable}_{dataset}_{date}.jpeg')
            plotMap(ds, variable, conf.coast_resolution, outName)

