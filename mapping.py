"""
Shared map helpers. Building a Basemap with intermediate/high resolution
coastlines takes seconds, so one instance is built per domain and reused
for every figure.
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

from utils import ticker

VAR_CONFIG = {
    'bias': dict(title='Bias [m]', vmin=-1.0, vmax=1.0, cmap='RdBu_r', extend='both'),
    'rmse': dict(title='RMSE [m]', vmin=0, vmax=1.0, cmap='YlOrRd', extend='max'),
    'mae': dict(title='MAE [m]', vmin=0, vmax=0.8, cmap='YlOrRd', extend='max'),
    'abs_error': dict(title='|model - sat| [m]', vmin=0, vmax=0.8, cmap='YlOrRd', extend='max'),
    'nbias': dict(title='NBIAS [%]', vmin=-60, vmax=60, cmap='RdBu_r', extend='both', scale=100),
    'nrmse': dict(title='NRMSE [%]', vmin=0, vmax=60, cmap='viridis', extend='max', scale=100),
    'si': dict(title='SI [%]', vmin=0, vmax=60, cmap='viridis', extend='max', scale=100),
    'correlation': dict(title='Correlation', vmin=0, vmax=1, cmap='RdYlGn', extend='min'),
    'nobs': dict(title='Number of observations', vmin=0, vmax=None, cmap='cividis', extend='max'),
}


def var_config(variable, values=None):
    base = variable.rsplit('_p', 1)[0] if variable not in VAR_CONFIG else variable
    if base in VAR_CONFIG:
        cfg = dict(VAR_CONFIG[base])
        if variable != base:
            cfg['title'] = f"{cfg['title']} (obs ≥ P{variable.rsplit('_p', 1)[1]})"
    else:
        finite = values[np.isfinite(values)] if values is not None else np.array([0, 1])
        cfg = dict(title=variable, vmin=np.percentile(finite, 5), vmax=np.percentile(finite, 95),
                   cmap='viridis', extend='both')
    cfg.setdefault('scale', 1)
    return cfg


class MapCanvas:
    def __init__(self, lon, lat, resolution='i', pad=0.25):
        self.lon_min, self.lon_max = float(np.nanmin(lon)) - pad, float(np.nanmax(lon)) + pad
        self.lat_min, self.lat_max = float(np.nanmin(lat)) - pad, float(np.nanmax(lat)) + pad
        self.m = Basemap(llcrnrlon=self.lon_min, llcrnrlat=self.lat_min,
                         urcrnrlon=self.lon_max, urcrnrlat=self.lat_max,
                         resolution=resolution, projection='merc')
        self.step = ticker(self.lat_min, self.lat_max)

    def new(self, figsize=(10, 9)):
        """Figure with the map background, drawn once and reused: the artists
        added for the previous plot are removed. Do not close the figure."""
        if getattr(self, 'fig', None) is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor='white')
            m, ax = self.m, self.ax
            m.drawmapboundary(fill_color='#DCEBF2', ax=ax)
            m.fillcontinents(color='#E0E0E0', lake_color='#DCEBF2', ax=ax, zorder=2)
            m.drawcoastlines(linewidth=0.8, color='#333333', ax=ax, zorder=3)
            m.drawparallels(np.arange(-90, 90 + self.step, self.step), labels=[1, 0, 0, 0],
                            linewidth=0.3, fontsize=10, color='gray', ax=ax)
            m.drawmeridians(np.arange(-180, 180 + self.step, self.step), labels=[0, 0, 0, 1],
                            linewidth=0.3, fontsize=10, color='gray', ax=ax)
            m.set_axes_limits(ax=ax)  # fixes the aspect before sizing the colorbar axis
            self.cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.15)
            self._base = set(ax.get_children())
        else:
            for artist in set(self.ax.get_children()) - self._base:
                artist.remove()
            self.cax.cla()
            self.ax.set_title('')
        return self.fig, self.ax

    def colorbar(self, fig, ax, mappable, label, extend='neither'):
        cb = fig.colorbar(mappable, cax=self.cax, extend=extend)
        cb.set_label(label, fontsize=12)
        return cb

    def save(self, outfile, dpi=200):
        self.fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor='white')

    @staticmethod
    def stats_box(ax, values, label='N'):
        v = values[np.isfinite(values)]
        if len(v) == 0:
            return
        text = f'{label} = {len(v)}\nμ = {v.mean():.3f}\nσ = {v.std():.3f}\nmin = {v.min():.3f}\nmax = {v.max():.3f}'
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'), zorder=10)
