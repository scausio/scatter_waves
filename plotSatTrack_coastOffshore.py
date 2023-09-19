import os
# import matplotlib
# matplotlib.use ('TkAgg')
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from scipy.spatial import Delaunay
import numpy as np
from datetime import datetime
from shapely.geometry import Point,LineString,Polygon
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def plot_map(name):
    sns.set_context("notebook")
    ax = plt.axes(projection=ccrs.PlateCarree())
    land = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                   edgecolor='face')
    #land=cartopy.feature.LAND('50m')
    ax.add_feature(land, color='lightgray', zorder=50)
    # ax.add_feature(cartopy.feature.COASTLINE)
    ax.set_extent([minLon, maxLon, minLat, maxLat])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False

    gl.xlocator = mticker.FixedLocator(np.arange(0, 90, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 1))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    print(str(ds_sat_coast.time.mean().values)[:19])
    date = datetime.strptime(str(ds_sat_coast.time.mean().values)[:19], '%Y-%m-%dT%H:%M:%S')
    title=f"2019-2020 CFOSAT tracks"
    plt.title(title, fontweight="bold")

    #ax.tricontourf(ds_ww3.longitude.values, ds_ww3.latitude.values, ds_ww3.tri.values-1,ds_ww3.hs.values, cmap='jet', vmin=0, vmax=8)
    im = ax.scatter(x_sat_c, y_sat_c, s=2, edgecolor='k', linewidth=0.2, color='crimson')
    im = ax.scatter(x_sat_os, y_sat_os, s=2, edgecolor='k', linewidth=0.2, color='chartreuse')
    #ax.plot(*sat_track.exterior.xy)
    plt.savefig(os.path.join(outpath,f"coast_offshore_track_map_{name}.png"))
    plt.show()
    plt.clf()


base_coast= '/work/opa/sc33616/ww3/tools/unst_Adr_2023_coast/adriatic_2019_2020_sat_series.nc'
base_os= '/work/opa/sc33616/ww3/tools/unst_Adr_2023_os/adriatic_2019_2020_sat_series.nc'
outpath='/work/opa/sc33616/ww3/tools/unst_Adr_2023/'
os.makedirs(outpath,exist_ok=True)

ds_sat_coast = xr.open_dataset(base_coast)
ds_sat_os = xr.open_dataset(base_os)

print (ds_sat_coast)

x_sat_c = ds_sat_coast.longitude
y_sat_c = ds_sat_coast.latitude

x_sat_os = ds_sat_os.longitude
y_sat_os = ds_sat_os.latitude


minLat=39#np.nanmin(y_sat)-0.5
maxLat=46#np.nanmax(y_sat)+0.5
minLon=12#np.nanmin(x_sat)-0.5
maxLon=21#np.nanmin(x_sat)+0.5


plot_map('adriatic')



