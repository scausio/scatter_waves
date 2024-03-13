import numpy as np
import xarray as xr
def BIAS(data, obs):
    return  np.round((np.nanmean( data-obs)).data, 4)


def RMSE(data,obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)),3)


def ScatterIndex(data,obs):
    num=np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom=np.sum(obs**2)
    return np.round(np.sqrt((num/denom)),3)


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
