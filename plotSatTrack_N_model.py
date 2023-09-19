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

    gl.xlocator = mticker.FixedLocator(np.arange(10, 40, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(30, 45, 1))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    print(str(ds_sat.time.mean().values)[:19])
    date = datetime.strptime(str(ds_sat.time.mean().values)[:19], '%Y-%m-%dT%H:%M:%S')
    title=f"{datetime.strftime(date, '%B %Y %d')} CFOSAT"
    plt.title(title, fontweight="bold")

    msk=np.isnan(ds_ww3_box.hs.values)
    ds_ww3_box.hs.values[msk]=10

    #ax.tricontourf(ds_ww3.longitude.values, ds_ww3.latitude.values, ds_ww3.tri.values-1,ds_ww3.hs.values, cmap='jet', vmin=0, vmax=8)
    tri=Delaunay(np.array((ds_ww3_box.longitude.values,ds_ww3_box.latitude.values)).T).simplices
    ax.tricontourf(ds_ww3_box.longitude.values, ds_ww3_box.latitude.values,tri ,ds_ww3_box.hs.values,np.linspace(vmin,vmax,30), cmap='jet')
    #ax.scatter(ds_ww3_box.longitude.values, ds_ww3_box.latitude.values, c= ds_ww3_box.hs.values,cmap='jet',
    #              )
    #ax.scatter(ds_ww3.longitude.values, ds_ww3.latitude.values, c=ds_ww3.hs.values, cmap='jet', vmin=0, vmax=8)

    im = ax.scatter(x_sat, y_sat, c=hs_sat, s=40, edgecolor='k', linewidth=0.2, cmap='jet', vmin=vmin, vmax=vmax)
    #ax.plot(*sat_track.exterior.xy)
    cb = plt.colorbar(im)
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('SWH [m]', rotation=270)
    plt.savefig(os.path.join(outpath,f"{title.replace(' ','_')}_map_{name}.png"))
    plt.show()
    plt.clf()


def plot_ts():
    sns.set_context("paper")
    ax=sns.lineplot(data=hs_sat,label='Sat',color='k',linestyle='dashed')
    sns.lineplot(data=ds_ww3_c.hs,label='WW3-Cou',color='r')
    sns.lineplot(data=ds_ww3_u.hs, label='WW3-Unc',color='b')
    o_hs =hs_sat
    mc_hs = ds_ww3_c.hs
    mu_hs = ds_ww3_u.hs
    print('Hs')
    BIAS_c=np.nanmean(o_hs-mc_hs)
    RMSE_c=np.round(np.sqrt(mean_squared_error(o_hs, mc_hs, squared=False)),3)
    corr_c, _ = pearsonr(o_hs, mc_hs)

    BIAS_u =np.round( np.nanmean(o_hs - mu_hs),3)
    RMSE_u=np.sqrt(mean_squared_error(o_hs, mu_hs, squared=False))
    corr_u, _ = pearsonr(o_hs, mu_hs)
    print('ro_u', corr_u)

    ax.set_ylabel('Hs [m]')
    ax.set_xticks([0,len(hs_sat)])
    ax.set_xticklabels([f'°N {np.round(y_sat.values[0],2)}, °E {np.round(x_sat.values[0],2)}', f'°N {np.round(y_sat.values[-1],2)}, °E {np.round(x_sat.values[-1],2)}'],fontsize=12)
    ax.legend()
    ax.set_ylim([0.5,5.5])
    date = datetime.strptime(str(ds_sat.time.mean().values)[:19], '%Y-%m-%dT%H:%M:%S')
    title=f"{datetime.strftime(date, '%Y %B %d')} - {ds_sat.platform}"
    plt.title(title, fontweight="bold")

    plt.text(0.3, 0.1, f'BIAS_c: {BIAS_c:.3f}\n'
                        f'RMSE_c: {np.round(RMSE_c,3):.3f}\n'
                        f'rho_c: {np.round(corr_c,3):.3f}\n'
                        f'BIAS_u: {np.round(BIAS_u,3):.3f}\n'
                        f'RMSE_u: {np.round(RMSE_u,3):.3f}\n'
                        f'rho_u: {np.round(corr_u,3):.3f}\n', transform=plt.gcf().transFigure)

    plt.savefig(os.path.join(outpath, f"{title.replace(' ','_')}_ts.png"))
    plt.show()
    plt.clf()


vmin=0
vmax=3.5


base='/work/opa/sc33616/ww3/tools/unst_Adr_2023/adriatic_2019_2020_sat_series.nc'
base_ww3='/work/opa/ww3_cst/runs/adriatic/20191201/netcdf/ww3.20201213.nc'
outpath='/work/opa/sc33616/ww3/tools/unst_Adr_2023/'
os.makedirs(outpath,exist_ok=True)

ds = xr.open_dataset(base)
ds_sat = ds.isel(obs=(ds.time.dt.month.isin(12)) & (ds.time.dt.day.isin(13))&(ds.time.dt.year.isin(2020)) )
print (ds_sat)
ds_ww3=xr.open_dataset(base_ww3).isel(time=17)
print(ds_ww3)
x_sat = ds_sat.longitude
y_sat = ds_sat.latitude

minLat=39#np.nanmin(y_sat)-0.5
maxLat=46#np.nanmax(y_sat)+0.5
minLon=12#np.nanmin(x_sat)-0.5
maxLon=21#np.nanmin(x_sat)+0.5



hs_sat = ds_sat.hs

sat_track = LineString([[x_sat[0], y_sat[0]], [x_sat[-1], y_sat[-1]]]).buffer(1.5)
# for i,base_ww3 in enumerate(base_ww3s):
#     name=['uncoupl','coupl']
#
#     ds_ww3=xr.open_dataset(base_ww3)
#     print (ds_ww3)
#     ds_ww3=ds_ww3.sel(time=ds_sat.time.mean(),method='nearest')
#
ww3_pnts=np.array((ds_ww3.longitude.values,ds_ww3.latitude.values)).T
ww3_points=list(map(Point,ww3_pnts))
box=[sat_track.contains(p) for p in ww3_points]

ds_ww3_box=ds_ww3.isel(node=box)
plot_map('adriatic')



