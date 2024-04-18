import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



f='/work/cmcc/now_rsc/validation/bs_azov/20210101_20221210_landMasked_qcheck_zscore6_ALLSAT.nc'
f='/work/cmcc/now_rsc/validation/bs_azov/20210101_20221210_series.nc'
f='/work/cmcc/ww3_cst-dev/tools/scatter_waves/output/Uglob2019/20190716_20190801_landMasked_qcheck_zscore6_ALLSAT.nc'
ds=xr.open_dataset(f)
print (ds)
x=ds.longitude.values
y=ds.latitude.values
print (x)
print (y)
print (np.nanmin(ds.hs.values),np.nanmax(ds.hs.values))
plt.plot(ds.hs)
#plt.colorbar()
print (1)
plt.show()
plt.savefig('./sattimeseries.png')

