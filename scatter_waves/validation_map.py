import matplotlib
matplotlib.use('Agg')
from stats import BIAS, RMSE, ScatterIndex
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, gaussian_kde
from utils import getConfigurationByID
import pandas as pd
from mpl_toolkits.basemap import Basemap

def metrics(data):
	result = xr.Dataset()
	bias = data['model_hs'] - data['hs']
	result['bias'] = bias.mean(dim='obs')
	obs_sum=data['hs'].sum(dim='obs')
	result['nbias'] = bias.sum(dim='obs')/obs_sum
	result['rmse'] =np.sqrt(bias ** 2.).mean(dim='obs')
	result['nrmse'] =np.sqrt((bias ** 2.).sum(dim='obs')/(data['hs']**2).sum(dim='obs'))
	result['nobs'] = bias.count(dim='obs')
	result['model_hs'] = data['model_hs'].mean(dim='obs')
	result['sat_hs'] = data['hs'].mean(dim='obs')
	return result

def coords2bins(ds_all,x,y,step):
	y_min, y_max = np.nanmin(ds_all.latitude.values), np.nanmax(ds_all.latitude.values)
	x_min, x_max = np.nanmin(ds_all.longitude.values), np.nanmax(ds_all.longitude.values)
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

def scatterPlot(sat, data, outname, **kwargs):
	print(sat.shape, data.shape, outname)

	xy = np.vstack([sat, data])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = sat[idx], data[idx], z[idx]
	fig, ax = plt.subplots()
	im = ax.scatter(x, y, c=z, s=3, cmap='jet')
	maxVal = np.max((x, y))

	ax.set_ylim(0, maxVal + 1)
	ax.set_xlim(0, maxVal + 1)
	ax.set_aspect(1.0)

	bias = BIAS(data, sat)
	corr, _ = pearsonr(x, y)
	rmse = RMSE(data, sat)
	si = ScatterIndex(data, sat)
	slope, intercept, rvalue, pvalue, stderr = linregress(sat, data)
	plt.text(0.8, 0.72, 'Entries: %s\n'
						'BIAS: %s\n'
						'RMSE: %s\n'
						'SI: %s\n'
						"$\\rho$:%s\n"
						'Slope: %s\n'
						'STDerr: %s'
			 % (len(x), bias, rmse, si, np.round(corr, 2),
				np.round(slope, 3), np.round(stderr, 3)), transform=plt.gcf().transFigure)

	if 'title' in kwargs:
		plt.title(kwargs['title'])

	if 'xlabel' in kwargs:
		plt.xlabel(kwargs['xlabel'])

	if 'ylabel' in kwargs:
		plt.ylabel(kwargs['ylabel'])

	sin = '+' if intercept >= 0 else ''
	print(intercept, 'intercept')
	plt.text(3, 0.25, 'y = {m}x {sin} {q}'.format(m=np.round(slope, 2), sin=sin, q=np.round(intercept, 2)),
			 horizontalalignment='center',
			 verticalalignment='top',
			 multialignment='center', size=9, style='italic')
	ax.plot([0, maxVal], [0, maxVal * slope], c='r')
	ax.plot([0, maxVal], [0, maxVal], c='k', linestyle='-.')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.colorbar(im, fraction=0.02)
	plt.savefig(outname, dpi=300)
	plt.close()

def maskNtimes(model, sat, times):
	diff = np.abs(model - sat)
	return diff > (model * times)


def plotMap(ds, variable, outfile):
	fig = plt.figure()
	fig.set_size_inches(8, 8)
	m = Basemap(llcrnrlon=ds.longitude.min(), llcrnrlat=ds.latitude.min(),
				urcrnrlat=ds.latitude.max(), urcrnrlon=ds.longitude.max(), resolution='l')
	m.drawcoastlines()
	m.fillcontinents('Whitesmoke')
	meridians = np.arange(-180, 190, 10)
	parallels = np.arange(-90,90, 10)
	m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
	m.drawmeridians(meridians, labels=[False, False, False, False], linewidth=0.1)
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
		plt.title(f'NRMSE [%]: {np.nanmean(ds[variable]*100)}', loc='left')
		vmin=0
		vmax=60
		cmap = 'jet'
		var=ds[variable]*100
	elif variable=='nbias':
		plt.title(f'NBias [%]: {np.nanmean(ds[variable]*100)}', loc='left')
		vmin=-60
		vmax=60
		cmap='seismic'
		var=ds[variable]*100

	im= plt.imshow(var, origin='lower', cmap=cmap, vmin=vmin,
						 vmax=vmax,
						 extent=[ds.longitude.min(), ds.longitude.max(), ds.latitude.min(), ds.latitude.max()])
	plt.colorbar(im)
	plt.savefig(outfile)
	plt.close()




# plt.show()

def main(conf_path, start_date, end_date):
	step = 0.1
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
		print(np.sum(np.isnan(sat_hs)))
		sat_hs = sat_hs.where(
			(ds_all.hs.values <= float(conf.filters.max)) & (ds_all.hs.values >= float(conf.filters.min)))
		model_hs = model_hs.where(
			(ds_all.model_hs.values <= float(conf.filters.max)) & (ds_all.model_hs.values >= float(conf.filters.min)))

		ds = get2Dbins(ds_all, step)

		print('saving output')
		#ds.to_netcdf('test_glob.nc')
		#for variable in ['bias','rmse']:
	#		outName = os.path.join(outdir, f'{variable}_{dataset}_{date}.jpeg')
	#		plotMap(ds, variable, outName)
		for variable in ['nbias','nrmse']:
                        outName = os.path.join(outdir, f'{variable}_{dataset}_{date}.jpeg')
                        plotMap(ds, variable, outName)

	# 	ds = {}
	# 	ds['sat'] = sat_hs.values
	# 	ds[dataset] = model_hs.values
	#
	# 	notValid = np.isnan(ds['sat'])
	#
	# 	print(len(ds['sat']))
	#
	# 	notValid = notValid | np.isnan(ds[dataset])
	#
	# 	print('sat-model valid ', np.where(~notValid))
	# 	if conf.filters.ntimes:
	# 		ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.filters.ntimes))
	# 		notValid = notValid | ntimes
	#
	# for dataset in conf.experiments:
	# 	outName = os.path.join(outdir, 'scatter_%s_%s.jpeg' % (dataset, date))
	# 	scatterPlot(ds['sat'][np.argwhere(~notValid)[:, 0]], ds[dataset][np.argwhere(~notValid)[:, 0]],
	# 				outName, title=f"{conf.title}".format(start_date=start_date, end_date=end_date),
	# 				xlabel=f'Sat SWH [m]', ylabel='Model SWH [m]')

