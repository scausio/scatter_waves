import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from stats import metrics
from utils import getConfigurationByID


def timeseries(ds,conf, outname, **kwargs):
    datasets=conf.experiments
    if 'title' in kwargs:
        plt.title(kwargs['title'])

    data_allmodels=ds.groupby('day').apply(metrics)
    print (data_allmodels)

    fig, ax = plt.subplots()
    colors=['r','b','m','o','c','g']
    for i,dataset in enumerate(datasets):
        data=data_allmodels.sel(model=dataset)
        ax.plot(data.day, data['model_hs'], c=colors[i], label=dataset.capitalize())
    
    if conf.filters.unbias in ['True','T','TRUE','t']:
        data['sat_hs']-=np.nanmean(data['sat_hs'])
        data['sat_hs']+=np.nanmean(data['model_hs'])
    ax.plot(data.day,data['sat_hs'],  c='k', label='satellite')
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
    for i,dataset in enumerate(datasets):
        data=data_allmodels.sel(model=dataset)

        ax1.plot(data.day,data['nrmse'], c=colors[i], label=f"{dataset.capitalize()}:{str(np.nanmean(data['nrmse']))[:5]}")
        ax2.plot(data.day,data['nbias'], c=colors[i], label=f"{dataset.capitalize()}:{str(np.nanmean(data['nbias']))[:6]}")
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


def main(conf_path,start_date,end_date):

    conf=getConfigurationByID(conf_path,'plot')
    outdir=os.path.join(conf.out_dir.out_dir,'plots')
    os.makedirs(outdir,exist_ok=True)
    date = f"{start_date}_{end_date}"

    for dataset in conf.experiments:
        ds_all=xr.open_dataset((conf.experiments[dataset].series).format(out_dir=conf.out_dir.out_dir,date=date,experiment=dataset))#.sel(model=dataset)
          

        ds_all['day'] = ('obs', ds_all.time.dt.floor('d').values)
        continue
    #for dataset in conf.experiments:
    outName=os.path.join(outdir, 'timeseriesAvg_%s_%s.jpeg' % (dataset,date))

        # sat=ds['sat'][np.argwhere(~notValid)[:, 0]]
        # model=ds[dataset][np.argwhere(~notValid)[:, 0]]
    print (ds_all)
    timeseries(ds_all,conf,
                   outName, title=f"{conf.title}".format(start_date=start_date,end_date=end_date))

