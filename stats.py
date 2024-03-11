import numpy as np
def BIAS(data, obs):
    return  np.round((np.nanmean( data-obs)).data, 4)


def RMSE(data,obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)),3)


def ScatterIndex(data,obs):
    num=np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom=np.sum(obs**2)
    return np.round(np.sqrt((num/denom)),3)
