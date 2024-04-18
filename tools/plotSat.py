import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



f='/work/cmcc/ww3_cst-dev/tools/scatter_waves/output/Uglob2019/20190701_20190731_landMasked_qcheck_zscore3_ALLSAT.nc'
ds=xr.open_dataset(f)
print (ds)
x=ds.longitude.values
y=ds.latitude.values
print (x)
print (y)
print (np.nanmin(ds.hs.values),np.nanmax(ds.hs.values))
plt.scatter(x,y, c=ds.hs)
#plt.colorbar()
print (1)
plt.show()
plt.savefig('./sattrack.png')

