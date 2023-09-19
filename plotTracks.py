import xarray as xr
import os
from glob import glob
from utils import getConfigurationByID
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

conf_pre = getConfigurationByID('.', 'sat_preproc')
conf = getConfigurationByID('.', 'plot')
outdir = conf.out_dir
os.makedirs(outdir,exist_ok=True)
years = f"{conf_pre.years[0]}_{conf_pre.years[-1]}" if len(conf_pre.years) > 1 else conf_pre.years[0]

# sat = (conf.sat.series).format(year=years,sigma=conf_pre.processing.filters.zscore.sigma)
# sat= xr.open_dataset(sat)

# create dataset
for dataset in conf.experiments:
	ds = xr.open_dataset((conf.experiments[dataset].series).format(year=years, experiment=dataset))
	print (ds)
	for mod in ds.model.values:

		outName = os.path.join(outdir, 'tracks_%s_%s.jpeg' % (mod, years))
		m = Basemap(llcrnrlon=np.nanmin(ds.longitude)-0.5, llcrnrlat=np.nanmin(ds.latitude)-0.5, urcrnrlat=np.nanmax(ds.latitude)+0.5, urcrnrlon=np.nanmax(ds.longitude)+0.5, resolution='i')
		m.drawcoastlines()
		m.fillcontinents('Whitesmoke')
		meridians = np.arange(-180,180,1)
		parallels = np.arange(-90,90,1)
		m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
		m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
		ds=ds.isel(obs=(ds.time.dt.month.isin(12))&(ds.time.dt.day.isin(13)))
		print (ds)
		# 	print (i)
		im = plt.scatter(ds.longitude,ds.latitude,c=ds['hs'],s=1,vmin=0,vmax=4,cmap='jet')
		plt.colorbar(im)
		plt.title('SWH [m]')
		plt.savefig(outName)
