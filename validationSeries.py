import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from scipy.stats import linregress, pearsonr, gaussian_kde
from utils import getConfigurationByID,checkOutdir
import math

def BIAS(data,obs):
    return  np.round((np.nanmean( data-obs)).data, 4)

def RMSE(data,obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)),3)

def ScatterIndex(data,obs):
    num=np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom=np.sum(obs**2)
    return np.round(np.sqrt((num/denom)),3)

def biasPlot(sat,model,expName, subset=False):

    plt.figure(figsize=(10,5))
    if not subset:
        plt.plot(sat - model)

        ticks_=np.linspace(0,len(labels)-1,5).astype(int)
        outName='%s_biasPlot.jpeg'%expName
    else:
        plt.plot(sat_hs[subset[0]:subset[1]] - model[subset[0]:subset[1]], 'g')
        plt.plot(sat_hs[subset[0]:subset[1]] - model[subset[0]:subset[1]], 'r')
        ticks_=np.linspace(subset[0],subset[1],5).astype(int)
        outName='%s_biasPlot_scat%s.jpeg'%(expName,subset[1])

    labels_=labels[ticks_]
    plt.xticks(ticks_,labels_,rotation=20)
    plt.legend(['WAM','WW3'])
    plt.ylabel('Bias')
    plt.savefig(os.path.join(outdir, outName),dpi=300)


def scatterPlot( sat, data, outname, **kwargs):
    print(sat.shape, data.shape, outname)

    xy = np.vstack([sat, data])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = sat[idx], data[idx], z[idx]
    fig, ax = plt.subplots()
    im = ax.scatter(x, y, c=z, s=3, edgecolor='', cmap='jet')
    maxVal = np.max((x, y))

    ax.set_ylim(0, maxVal+1)
    ax.set_xlim(0, maxVal+1)
    ax.set_aspect(1.0)
    
    bias = BIAS (data,sat)
    corr, _ = pearsonr(x, y)
    rmse=RMSE(data,sat)
    si=ScatterIndex(data,sat)
    slope,intercept, rvalue,pvalue,stderr=linregress( data,sat)
    plt.text(0.8, 0.72, 'Entries: %s\n'
             'BIAS: %s\n'
             'RMSE: %s\n'
             'SI: %s\n'
             "$\\rho$:%s\n"
             'Slope: %s\n'
             'STDerr: %s'
             %(len(x),bias,rmse,si,np.round(corr,2),
               np.round(1+(1-slope),3),np.round(stderr,3)),transform=plt.gcf().transFigure)

    if 'title' in kwargs:
        plt.title(kwargs['title'])



    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'])

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'])

    plt.text(3,0.25, 'y = {m}x + {q}'.format(m=np.round(1+(1-slope),2),q=np.round(intercept,2)),
         horizontalalignment='center',
         verticalalignment='top',
         multialignment='center',size=9,style='italic')


    ax.plot([0,np.max(data)],[0,np.max(data)*1+(1-slope)], c='r')
    ax.plot([0,np.max(data)],[0,np.max(data)],c='k',linestyle='-.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.colorbar(im,fraction=0.02)
    plt.savefig(outname, dpi=300)
    plt.close()


def maskNtimes(model,sat,times):
     diff=np.abs(model-sat)
     return diff>(model*times) 

def dateParser(timeObj):
    return datetime.datetime.strptime((':').join(str(timeObj).split(':')[:2]).replace('T', '-'), '%Y-%m-%d-%H:%M')


def main():
    conf_pre=getConfigurationByID('.','sat_preproc')
    conf=getConfigurationByID('.','plot')
    outdir=conf.out_dir
    checkOutdir(conf.out_dir)

    years=f"{conf_pre.years[0]}_{conf_pre.years[-1]}" if len(conf_pre.years)>1  else conf_pre.years[0]

    sat = (conf.sat.series).format(year=years,sigma=conf_pre.processing.filters.zscore.sigma)
    sat= xr.open_dataset(sat)
    print (sat)
    sat_hs=sat[conf.sat.hs]

    ds={}
    ds['sat']=sat_hs.values
    notValid=np.isnan(ds['sat'])
    print(np.where( ~notValid))
    # create dataset
    for dataset in conf.experiments:
        ds[dataset]= np.load((conf.experiments[dataset].series).format(year=years,experiment=dataset))
        notValid=notValid | np.isnan(ds[dataset])
        print(np.where( ~notValid))
        if conf.additional_filters.ntimes:
            ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.additional_filters.ntimes))
            notValid = notValid | ntimes

    # mask nans in all datasets
    #print(np.where( ~notValid))
    #for dataset in conf.experiments:
    #    ds.update({dataset:ds[dataset][~notValid]})
    #    print (ds[dataset].shape)
    #    print(len( notValid))
    print (np.argwhere(np.isnan(ds['sat'][ np.argwhere(~notValid)])))
    #print (notValid)
    # scatter plot
    for dataset in conf.experiments:
        print (np.argwhere(np.isnan(ds[dataset][ np.argwhere(~notValid)])))
        outName=os.path.join(outdir, 'scatter_%s_%s.jpeg' % (dataset,years))
        #scatterPlot(ds['sat'][ np.argwhere(~notValid)][:,0], ds[dataset][ np.argwhere(~notValid)][:,0], outName, title=f"{years} {dataset}",xlabel='%s Hs[m]'%conf.satName,ylabel='%s Hs[m]'%dataset)
        scatterPlot(ds['sat'][ np.argwhere(~notValid)][:,0], ds[dataset][ np.argwhere(~notValid)][:,0], outName, title=years.replace('_','-'),xlabel=f'{conf.satName}',ylabel='Model SWH [m]')

if __name__ == '__main__':
    main()
