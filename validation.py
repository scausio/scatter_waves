import matplotlib
matplotlib.use('Agg')
from stats import BIAS, RMSE, ScatterIndex
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, gaussian_kde
from utils import getConfigurationByID

def scatterPlot( sat, data, outname, **kwargs):
    print(sat.shape, data.shape, outname)

    xy = np.vstack([sat, data])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = sat[idx], data[idx], z[idx]
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=z, s=3, cmap='jet')
    maxVal = np.max((np.max(x), np.max(y)))

    ax.set_ylim(0, maxVal+1)
    ax.set_xlim(0, maxVal+1)
    ax.set_aspect(1.0)
    
    bias = BIAS(data, sat)
    corr, _ = pearsonr(x, y)
    rmse= RMSE(data, sat)
    si= ScatterIndex(data, sat)
    slope,intercept, rvalue,pvalue,stderr=linregress( sat,data)
    plt.text(0.8, 0.72, 'Entries: %s\n'
             'BIAS: %s\n'
             'RMSE: %s\n'
             'SI: %s\n'
             "$\\rho$:%s\n"
             'Slope: %s\n'
             'STDerr: %s'
             %(len(x),bias,rmse,si,np.round(corr,2),
               np.round(slope,3),np.round(stderr,3)),transform=plt.gcf().transFigure)
    
    if 'title' in kwargs:
        plt.title(kwargs['title'])

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])

    sin = '+'if intercept>=0 else ''
    print (intercept, 'intercept')
    plt.text(3,0.25, 'y = {m}x {sin} {q}'.format(m=np.round(slope,2),sin=sin,q=np.round(intercept,2)),
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center',size=9,style='italic')
    ax.plot([0,maxVal],[0,maxVal*slope], c='r')
    ax.plot([0,maxVal],[0,maxVal],c='k',linestyle='-.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.colorbar(im,fraction=0.02)
    plt.savefig(outname, dpi=300)
    plt.close()

def maskNtimes(model,sat,times):
     diff=np.abs(model-sat)
     print ('diff',np.nanmax(diff))
     print ('mod-tim',np.nanmax(model*times))
     print ('times',np.nanmax(times))
     return diff>(model*times) 

def main(conf_path,start_date,end_date):

    conf=getConfigurationByID(conf_path,'plot')
    outdir=os.path.join(conf.out_dir.out_dir,'plots')
    os.makedirs(outdir,exist_ok=True)
    date = f"{start_date}_{end_date}"
    ds={}
    for dataset in conf.experiments:
        ds_all=xr.open_dataset((conf.experiments[dataset].series).format(out_dir=conf.out_dir.out_dir,date=date))
        print (dataset)
        print (ds_all)
        sat_hs = ds_all.hs
        model_hs = ds_all.model_hs
        print (model_hs)
        model_hs = model_hs.where(~np.any(np.isnan(model_hs), axis=1), np.nan)

        print (np.sum (np.isnan(sat_hs)))
        print ('max_sat:',np.nanmax(ds_all.hs.values))
        print ('min_sat:',np.nanmin(ds_all.hs.values))
        sat_hs=sat_hs.where(
            (ds_all.hs.values <= float(conf.filters.max)) & (ds_all.hs.values >= float(conf.filters.min)))
        model_hs=model_hs.where(
            (ds_all.model_hs.values <= float(conf.filters.max)) & (ds_all.model_hs.values >= float(conf.filters.min)))
        
        
        ds['sat'] = sat_hs.values
        ds[dataset] = model_hs.sel(model=dataset).values

        notValid=np.isnan(ds['sat'])

        print (len(ds['sat']),ds[dataset].shape)

        notValid=notValid | np.isnan(ds[dataset])

        print('sat-model valid ',np.where( ~notValid))
        if conf.filters.ntimes:
            ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.filters.ntimes))
            notValid = notValid | ntimes

    for dataset in conf.experiments:
        outName=os.path.join(outdir, 'scatter_%s_%s.jpeg' % (dataset,date))
        sat2plot=ds['sat'][ np.argwhere(~notValid)[:,0]]
        mod2plot=ds[dataset][ np.argwhere(~notValid)[:,0]]
        if conf.filters.unbias in ['True','T','TRUE','t']:
            sat2plot-=np.nanmean(sat2plot)
            sat2plot+=np.nanmean(mod2plot)
        scatterPlot(sat2plot,mod2plot,
                    outName, title=f"{conf.title} - {dataset.capitalize()}".format(start_date=start_date,end_date=end_date),xlabel=f'Sat SWH [m]',ylabel='Model SWH [m]')

