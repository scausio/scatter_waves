import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
from stats import BIAS, RMSE, ScatterIndex
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, gaussian_kde
from utils import getConfigurationByID

def scatter_waves(model_data, satellite_data, 
                                 model_name="Model", sat_name="Satellite",
                                 figsize=(12, 8), save_path=None, dpi=300):
    """
    
    Parameters:
    -----------
    model_data : array-like
    satellite_data : array-like  
    model_name : str
    sat_name : str
    figsize : tuple
    save_path : str, optional
    dpi : int
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    valid_mask = (~np.isnan(model_data)) & (~np.isnan(satellite_data)) & \
                 (model_data >= 0) & (satellite_data >= 0)
    model_clean = np.array(model_data)[valid_mask]
    sat_clean = np.array(satellite_data)[valid_mask]
    if len(model_clean) == 0:
        raise ValueError("No data available")
    
    bias = np.mean(model_clean - sat_clean)
    rmse = np.sqrt(np.mean((model_clean - sat_clean)**2))
    mae = np.mean(np.abs(model_clean - sat_clean))
    correlation, p_value = stats.pearsonr(model_clean, sat_clean)
    r2 = r2_score(sat_clean, model_clean)
    
    # Normalized stats
    mean_obs = np.mean(sat_clean)
    normalized_bias = bias / mean_obs * 100 if mean_obs > 0 else 0
    normalized_rmse = rmse / mean_obs * 100 if mean_obs > 0 else 0
    scatter_index = ScatterIndex(model_clean,sat_clean)*100
    #plt.style.use('default')
    sns.set_palette(sns.color_palette("plasma"))

    # Creazione figura con layout personalizzato
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Layout con proporzioni personalizzate
    gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 2, 0.8], width_ratios=[2, 0.8, 0.8],
                         hspace=0.3, wspace=0.3)
    
    # Titolo principale
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, f'{model_name} vs {sat_name}', 
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 color='#2C3E50', transform=title_ax.transAxes)
    title_ax.axis('off')
    
    # Plot principale - Scatter
    ax_main = fig.add_subplot(gs[1, 0])
    
    max_val = max(np.max(sat_clean), np.max(model_clean))
    buffer = (max_val ) * 0.2
    plot_min = max(0, 0)
    plot_max = max_val + buffer
    
    if len(model_clean) > 1000:
        # Hexbin per grandi dataset
        scatter = ax_main.hexbin(sat_clean, model_clean, gridsize=100, cmap='plasma', 
                           mincnt=1, alpha=0.8, extent=[plot_min, plot_max, plot_min, plot_max])
    else:
        xy = np.vstack([sat_clean, model_clean])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = sat_clean[idx], model_clean[idx], z[idx]
        scatter = ax_main.scatter(x,y, c=z, alpha=0.7,cmap='plasma', 
                                 s=40, edgecolors='white', linewidth=0.5)
    cb = plt.colorbar(scatter, ax=ax_main, pad=0.02, shrink=0.8)
    cb.set_label('Density', fontsize=11, labelpad=10)
    ax_main.plot([0, plot_max], [0, plot_max], 'r--', 
                linewidth=2, label='Best fit', alpha=0.8)
    
    # Linea di regressione
    slope, intercept, r_value, p_val, std_err = stats.linregress(sat_clean, model_clean)
    line_x = np.array([0, plot_max])
    line_y =slope * line_x + intercept
    ax_main.plot(line_x, line_y, 'gray', linewidth=2, alpha=0.9,
                label=f'Regression (y={slope:.2f}x+{intercept:.2f})')
    
    # Bande di confidenza (±RMSE)
    #ax_main.fill_between([plot_min, plot_max], 
    #                    [plot_min - rmse, plot_max - rmse],
    #                    [plot_min + rmse, plot_max + rmse], 
    #                    alpha=0.2, color='orange', label=f'±RMSE ({rmse:.3f}m)')
    
    # Personalizzazione assi principali
    ax_main.set_xlabel(f'{sat_name}', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f'Model SWH [m]', fontsize=14, fontweight='bold')
    ax_main.set_xlim(0, plot_max)
    ax_main.set_ylim(0, plot_max)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_main.legend(loc='upper left', frameon=True, fancybox=True)
    
    # Box con statistiche principali nel plot
    stats_text = f'N = {len(model_clean):,}\n'
    stats_text += f'R² = {r2:.3f}\n'
    stats_text += f'Correlazione = {correlation:.3f}\n'
    stats_text += f'RMSE = {rmse:.3f} m\n'
    stats_text += f'Bias = {bias:.3f} m'
    
    # Box delle statistiche con sfondo elegante
    #bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightblue", 
    #                 alpha=0.8, edgecolor="navy", linewidth=1.5)
    #ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
    #            fontsize=11, verticalalignment='top', fontweight='bold',
    #            bbox=bbox_props)
    
    
    # Plot marginale superiore (distribuzione satellite)
    ax_top = fig.add_subplot(gs[1, 1])
    ax_top.hist(sat_clean, bins=30, alpha=0.7, color='skyblue',
            edgecolor='black',density=True)
    ax_top.set_xlim(plot_min, plot_max)  # sync X axis to data range
    ax_top.set_title(f'Distribution\n{sat_name}', fontsize=10, fontweight='bold')
    ax_top.grid(True, alpha=0.3)

    # Plot marginale destro (distribuzione modello)
    ax_right = fig.add_subplot(gs[1, 2])
    countsT, _ = np.histogram(sat_clean, bins=30, density=True)
    countsR, _ = np.histogram(model_clean, bins=30, density=True)
    max_density = max(np.max(countsT), np.max(countsR))
    ax_top.set_ylim(plot_min, max_density * 1.1)  # small buffer
    ax_right.hist(model_clean, bins=30, alpha=0.7, color='lightcoral',
              edgecolor='black',density=True)
    ax_right.set_xlim(plot_min, plot_max)
    ax_right.set_ylim(plot_min, max_density * 1.1)
    ax_right.set_title(f'Distribution\n{model_name}', fontsize=10, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    # Tabella delle statistiche dettagliate
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Preparazione dati per la tabella
    stats_data = [
        ['Metrics', 'Value', 'Description'],
        ['Entries', f'{len(model_clean):,}', 'Valid observations'],
        ['ρ', f"{correlation:.4f}", "Pearson's correlation coefficient"],
        ['R²', f'{r2:.4f}', 'Coefficiente di determinazione'],
        ['RMSE (m)', f'{rmse:.4f}', 'Root Mean Square Error'],
        ['MAE (m)', f'{mae:.4f}', 'Mean Absolute Error'],
        ['Bias (m)', f'{bias:.4f}', 'Averaged sistematic error'],
        ['NRMSE (%)', f'{normalized_rmse:.2f}', 'normalized RMSE'],
        ['NBias (%)', f'{normalized_bias:.2f}', 'normalized Bias'],
        ['SI (%)', f'{scatter_index:.2f}', 'Scatter Index'],
        ['Avg Satellite (m)', f'{mean_obs:.4f}', 'Satellite mean value'],
        ['Avg Model (m)', f'{np.mean(model_clean):.4f}', 'Model mean value']
    ]
    
    # Creazione tabella con stile
    table = ax_stats.table(cellText=stats_data[1:], colLabels=stats_data[0],
                          cellLoc='center', loc='center', colWidths=[0.2, 0.15, 0.65], bbox=[0.0, -0.25, 1, 1.2] )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.8, 1.8)

    for i in range(len(stats_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#ECF0F1')
                else:
                    cell.set_facecolor('white')
                    
                #if j == 1:  # Colonna valori
                #    if 'R²' in stats_data[i][0] and float(stats_data[i][1]) < 0.5:
                #        cell.set_facecolor('#FFE6E6')  # Rosso chiaro per R² bassi
                #    elif 'RMSE' in stats_data[i][0]:
                #        cell.set_facecolor('#E6F3FF')  # Blu chiaro per RMSE
    
   # fig.text(0.99, 0.01, 'Validazione Modello Onde', ha='right', va='bottom', 
   #          alpha=0.3, fontsize=8, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Plot saved in: {save_path}")
    
    return fig



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

    for i,dataset in enumerate(conf.experiments):
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
        if i==0:
            notValid=np.isnan(ds['sat'])

        print (len(ds['sat']),ds[dataset].shape)

        notValid=notValid | np.isnan(ds[dataset])

        print('sat-model valid ',np.where( ~notValid))
        if conf.filters.ntimes:
            ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.filters.ntimes))
            notValid = notValid | ntimes

    sat2plot=ds['sat'][ np.argwhere(~notValid)[:,0]]


    for i,dataset in enumerate(conf.experiments):
        outName=os.path.join(outdir, 'scatter_%s_%s.jpeg' % (dataset,date))
        mod2plot=ds[dataset][ np.argwhere(~notValid)[:,0]]
        if (conf.filters.unbias in ['True','T','TRUE','t']) & (i==0):
            sat2plot-=np.nanmean(sat2plot)
            sat2plot+=np.nanmean(mod2plot)
        fig = scatter_waves(
        mod2plot, sat2plot,
        model_name=f"{dataset} SWH [m]",
        sat_name="Sat SWH [m]",
        save_path=outName)
    

