import matplotlib
matplotlib.use('Agg')
from stats import BIAS, RMSE, ScatterIndex
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from stats import metrics
from utils import getConfigurationByID


def timeseries(ds, outname, **kwargs):

    if 'title' in kwargs:
        plt.title(kwargs['title'])

    data=ds.groupby('day').apply(metrics)

    print (data)
    fig, ax = plt.subplots()
    ax.plot(data.day,data['sat_hs'],  c='k', label='satellite')
    ax.plot(data.day, data['model_hs'], c='r', label='model')
    plt.ylabel('SWH [m]')
    plt.xticks(rotation=30)
    #
    plt.grid(linewidth=0.3)
    plt.legend()
    plt.savefig(outname, dpi=200)
    plt.close()

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'])

    ax1.plot(data.day,data['nrmse'], c='k', label=f"NRMSE:{str(np.nanmean(data['nrmse']))[:5]}")
    ax2.plot(data.day,data['nbias'], c='k', label=f"NBIAS:{str(np.nanmean(data['nrmse']))[:6]}")
    ax1.set_ylabel('NRMSE [m]')
    ax2.set_ylabel('NBIAS [m]')

    plt.xlabel('')
    plt.xticks(rotation=30)

    ax1.grid(linewidth=0.3)
    ax2.grid(linewidth=0.3)
    ax1.legend()
    ax2.legend()

    ax1.set_ylim(ymin=-0.5, ymax=1.5)
    ax2.set_ylim(ymin=-1.5, ymax=1.5)

    plt.savefig(outname.replace('Avg','_stats_'), dpi=200)
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

    for dataset in conf.experiments:
        print (dataset)
        ds_all=xr.open_dataset((conf.experiments[dataset].series).format(out_dir=conf.out_dir.out_dir,date=date,experiment=dataset)).sel(model=dataset)
        ds_all['day'] = ('obs', ds_all.time.dt.floor('d').values)
        #sat_hs = ds_all.hs
        #model_hs = ds_all.model_hs

        # print (np.sum (np.isnan(sat_hs)))
        # print ('max_sat:',np.nanmax(ds_all.hs.values))
        # print ('min_sat:',np.nanmin(ds_all.hs.values))
        # sat_hs=sat_hs.where(
        #     (ds_all.hs.values <= float(conf.filters.max)) & (ds_all.hs.values >= float(conf.filters.min)))
        # model_hs=model_hs.where(
        #     (ds_all.model_hs.values <= float(conf.filters.max)) & (ds_all.model_hs.values >= float(conf.filters.min)))
        #
        # ds = {}
        # ds['sat'] = sat_hs.values
        # ds[dataset] = model_hs.values
        #
        # notValid=np.isnan(ds['sat'])
        #
        # print (len(ds['sat']))
        #
        # notValid=notValid | np.isnan(ds[dataset])
        #
        # print('sat-model valid ',np.where( ~notValid))
        # if conf.filters.ntimes:
        #     ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.filters.ntimes))
        #     notValid = notValid | ntimes

    for dataset in conf.experiments:
        outName=os.path.join(outdir, 'timeseriesAvg_%s_%s.jpeg' % (dataset,date))

        # sat=ds['sat'][np.argwhere(~notValid)[:, 0]]
        # model=ds[dataset][np.argwhere(~notValid)[:, 0]]
        print (ds_all)
        timeseries(ds_all,
                   outName, title=f"{conf.title}".format(start_date=start_date,end_date=end_date))

