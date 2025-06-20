import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from stats import metrics
from utils import getConfigurationByID
import seaborn as sns
from matplotlib.dates import DateFormatter

def timeseries(ds, conf, outname, **kwargs):
    # Set modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    datasets = conf.experiments

    data_allmodels = ds.groupby('day').apply(metrics)
    print(data_allmodels)

    # Enhanced color palette - modern and visually appealing
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    # FIRST PLOT: Main timeseries with enhanced styling
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Plot model data with enhanced styling
    for i, dataset in enumerate(datasets):
        data = data_allmodels.sel(model=dataset)
        ax.plot(data.day, data['model_hs'],
                color=colors[i],
                label=dataset.capitalize(),
                linewidth=2.5,
                alpha=0.8,
                marker='o',
                markersize=4,
                markevery=max(1, len(data.day) // 20))  # Show markers but not too dense

    # Handle bias correction
    if conf.filters.unbias in ['True', 'T', 'TRUE', 't']:
        data['sat_hs'] -= np.nanmean(data['sat_hs'])
        data['sat_hs'] += np.nanmean(data['model_hs'])

    # Plot satellite data with enhanced styling
    ax.plot(data.day, data['sat_hs'],
            color='#2C3E50',
            label='Satellite',
            linewidth=3,
            alpha=0.9,
            linestyle='-',
            marker='s',
            markersize=3,
            markevery=max(1, len(data.day) // 25))

    # Enhanced styling
    ax.set_ylabel('Significant Wave Height [m]', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize=16, fontweight='bold', pad=20)

    # Improve grid
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)

    # Enhanced legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                       shadow=True, fontsize=11, framealpha=0.9)
    legend.get_frame().set_facecolor('white')

    # Format x-axis dates
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(outname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # SECOND PLOT: Statistics subplot with enhanced styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('white')

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=16, fontweight='bold', y=0.95)

    # Set background colors
    ax1.set_facecolor('#FAFAFA')
    ax2.set_facecolor('#FAFAFA')

    # Plot statistics with enhanced styling
    for i, dataset in enumerate(datasets):
        data = data_allmodels.sel(model=dataset)

        # NRMSE plot
        mean_nrmse = np.nanmean(data['nrmse'])
        ax1.plot(data.day, data['nrmse'],
                 color=colors[i],
                 label=f"{dataset.capitalize()}: {mean_nrmse:.3f}",
                 linewidth=2.5,
                 alpha=0.8,
                 marker='o',
                 markersize=3,
                 markevery=max(1, len(data.day) // 20))

        # NBIAS plot
        mean_nbias = np.nanmean(data['nbias'])
        ax2.plot(data.day, data['nbias'],
                 color=colors[i],
                 label=f"{dataset.capitalize()}: {mean_nbias:.3f}",
                 linewidth=2.5,
                 alpha=0.8,
                 marker='^',
                 markersize=3,
                 markevery=max(1, len(data.day) // 20))

    # Enhanced styling for both subplots
    ax1.set_ylabel('NRMSE [m]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('NBIAS [m]', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=14, fontweight='bold')

    # Add horizontal reference lines
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Enhanced grids
    ax1.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax2.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # Enhanced legends
    legend1 = ax1.legend(loc='upper left', frameon=True, fancybox=True,
                         shadow=True, fontsize=10, framealpha=0.9)
    legend2 = ax2.legend(loc='upper left', frameon=True, fancybox=True,
                         shadow=True, fontsize=10, framealpha=0.9)

    legend1.get_frame().set_facecolor('white')
    legend2.get_frame().set_facecolor('white')

    # Set axis limits with some padding
    ax1.set_ylim(ymin=-0.6, ymax=1.6)
    ax2.set_ylim(ymin=-1.6, ymax=1.6)

    # Format ticks
    ax1.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.tick_params(axis='x', rotation=45)

    # Add subtle borders
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    plt.savefig(outname.replace('Avg', '_stats_'), dpi=300, bbox_inches='tight', facecolor='white')
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

