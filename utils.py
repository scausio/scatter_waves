import os
from datetime import date, timedelta

import munch
import yaml


def getConfigurationByID(path, confId):
    with open(path) as f:
        globalConf = yaml.load(f, Loader=yaml.Loader)
    return munch.Munch.fromDict(globalConf[confId])


def daysBetweenDates(start_date, end_date):
    start_date = str(start_date)
    end_date = str(end_date)

    start_date = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:]))
    end_date = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:]))
    delta = end_date - start_date
    return [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]


def ticker(coord_min, coord_max):
    dif = coord_max - coord_min
    if dif >= 180:
        step = 25
    elif dif >= 50:
        step = 10
    elif dif >= 25:
        step = 5
    elif dif >= 10:
        step = 2
    else:
        step = 1
    return step


def as_bool(value):
    """YAML may give a real bool or a string such as 'True'/'T'/'false'."""
    if isinstance(value, str):
        return value.strip().lower() in ('true', 't', 'yes', 'y', '1')
    return bool(value)


def n_workers(requested=None):
    """Number of worker processes: CLI value, else LSF allocation, else 1."""
    if requested:
        return max(1, int(requested))
    lsf = os.environ.get('LSB_DJOB_NUMPROC')
    if lsf:
        return max(1, int(lsf))
    return 1
