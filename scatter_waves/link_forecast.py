import os
from calendar import monthrange

def submit(base,out,years,months):
    os.makedirs(out, exist_ok=True)
    if not isinstance(years, list):
        years=[years]
    if not isinstance(months, list):
        months = [months]
    for year in years:
        for month in months:
            dayofmonth = monthrange(year, month)[1]
            for day in range(1, dayofmonth + 1):
                date = f'{year}{month:02d}{day:02d}'
                fl = f'ww3.{date}.nc'
                if not os.path.islink(os.path.join(out, fl)):
                    print (f'linking {os.path.join(out, fl)}')
                    os.symlink(os.path.join(base.format(date=date),date, fl), os.path.join(out, fl))
if __name__=="__main__":
    years=2023
    months=9
    base='/data/opa/ww3_cst/essenceWav/ww3/adri/'
    out='/work/opa/sc33616/ww3/tools/iride/adri/'
    submit(base,out,years,months)
