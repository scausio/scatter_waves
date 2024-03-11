

conf=getConfigurationByID('.','sat_preproc')
base=conf.paths.output
fname_tmpl=f'*_threshold_landMasked_qcheck_zscore{conf.processing.filters.zscore.sigma}.nc'

ff=glob(os.path.join(base,fname_tmpl))
#xr.open_mfdataset(ff, combine='by_coords',concat_dim='obs')
mrgd=myMFdataset(ff,conf)
print (mrgd)
years=f"{conf.years[0]}_{conf.years[-1]}" if len(conf.years)>1  else conf.years[0]
mrgd.to_netcdf(os.path.join(base,'{yy}_{tmpl}_ALLSAT.nc'.format(yy=years,tmpl=fname_tmpl[2:-3])))
