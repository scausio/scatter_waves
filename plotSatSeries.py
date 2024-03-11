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

    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 10))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    print(str(ds_sat_coast.time.mean().values)[:19])

    title=f"Sat tracks"
    plt.title(title, fontweight="bold")

    #ax.tricontourf(ds_ww3.longitude.values, ds_ww3.latitude.values, ds_ww3.tri.values-1,ds_ww3.hs.values, cmap='jet', vmin=0, vmax=8)
    im = ax.scatter(x_sat, y_sat, s=2, edgecolor='k', linewidth=0.2, color='crimson')
    #ax.plot(*sat_track.exterior.xy)
    plt.savefig(os.path.join(outpath,f"track_map_{name}.png"))
    plt.show()
    plt.clf()

base= '/work/opa/sc33616/ww3/tools/unst_Adr_2023_os/adriatic_2019_2020_sat_series.nc'
outpath='/work/opa/sc33616/ww3/tools/unst_Adr_2023/'
os.makedirs(outpath,exist_ok=True)

ds_sat = xr.open_dataset(base)
print (ds_sat)


x_sat = ds_sat.longitude
y_sat = ds_sat.latitude


minLat=np.nanmin(y_sat)-0.5
maxLat=np.nanmax(y_sat)+0.5
minLon=np.nanmin(x_sat)-0.5
maxLon=np.nanmin(x_sat)+0.5

plot_map('adriatic')



