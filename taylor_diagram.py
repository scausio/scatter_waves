#!/usr/bin/env python
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
import seaborn as sns
from utils import getConfigurationByID
sns.set_context('notebook')

def maskNtimes(model,sat,times):
     return np.array(sat>(model*times)) | np.array(model>(sat*times))

def getValid(obs,models,maskTimes=2):
    obs[obs<0.25]=np.nan
    obs[obs>10]=np.nan

    for exp in models.keys():
        exp_data=models[exp]
        ntimes=maskNtimes(exp_data,obs,maskTimes)
        obs[ntimes]=np.nan
        obsNan=np.isnan(obs)
 
        exp_data[ntimes]=np.nan

        exp_data[exp_data<0.25]=np.nan
        expNan=np.isnan(exp_data)
        
        obs[obsNan+expNan]=np.nan

    return np.logical_not(np.isnan(obs))

class TaylorDiagram(object):
    """
    Taylor diagram.

    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters:

        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """



        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)    # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1 ,grid_locator2=gf.MaxNLocator(5),  tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()
        
        ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")


        ax.axis["left"].toggle(ticklabels=False, label=True)
        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=5, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours



def applyTaylor(obs,models,year,exps):
    """Display a Taylor diagram in a separate axis."""

    # Reference dataset
    refstd = obs.std(ddof=1)           # Reference standard deviation

    samples = np.array([[m.std(ddof=1), np.corrcoef(obs, m)[0, 1]]
                        for m in models])

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1, xlabel='Time', ylabel='Hs [m]')
    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, rect=122, label="Obs",
                        srange=(0.3, 1.5))

    x=np.arange(len(obs))
    ax1.plot(x, obs, 'ko', label='Obs',markersize=5)
    for i, m in enumerate(models):
        ax1.plot(x, m, label=exps[i])
        #ax1.plot(x, m, c=colors[i], label='Model %d' % (i+1))
    ax1.legend(numpoints=1, prop=dict(size='small'), loc='best')
    ax1.grid(alpha=0.7, linestyle='dotted')
    # Add the models to Taylor diagram
    for i, (stddev, corrcoef) in enumerate(samples):
        dia.add_sample(stddev, corrcoef,marker='o', ms=5, ls='',
                       #marker='$%d$' % (i+1), ms=10, ls='',
                       #mfc=colors[i], mec=colors[i],
                       label=exps[i])

    # Add grid
    dia.add_grid(alpha=0.7,linestyle='dotted')

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

    # Add a figure legend
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')
    plt.tight_layout()
    if isinstance(year, list):
        year=f"{year[0]}_{year[-1]}"
    else:
        pass
    plt.savefig(os.path.join(outdir,f'taylor_diag_{year}_{exp}.png' ))
    plt.close()

def main():
    outdir='/work/opa/now_rsc/ww3/bs_validation/taylor_diag'
    years=[2016,2017,2018]
    exps=[ 'wam','BS_ERA5_test_uv_rea', 'BS_ERA5_test_uvT_rea','BS_ERA5_test_uvTl_rea','BS_ERA5_test_unco_rea','bs-ww3_v1.0_ts']
    mask_n_times=3

    os.makedirs(outdir,exist_ok=True)
    sat_buffer=[]
    model_buffer=[]

    for year in years:
        print (year)
        sat_path = f'/work/opa/now_rsc/ww3/bs_validation/data/{year}_j2_blackSea_zscore3_maskLandBuf20km.nc'
        model={}
        for exp in exps:
            print (exp)
            if exp=='wam':
                exp_path=f'/work/opa/now_rsc/ww3/bs_validation/data/wam_{year}_series.npy'
            else:
                exp_path =f'/work/opa/now_rsc/ww3/bs_validation/data/ww3_{exp}_{year}_series.npy'

            model[exp]=np.load(exp_path)

        sat= xr.open_dataset(sat_path).swh_ku.values
        print(f'{np.sum(np.isfinite(sat))} satellite occurrences before filtering')

        filt=getValid(sat, model ,mask_n_times)
        obs=sat[filt]
        print(f'{len(obs)} satellite occurrences after filtering')
        models=[v[filt] for k,v in model.items()]
        [sat_buffer.append(i) for i in obs]
        applyTaylor(obs,models, year,exps)
        exp_dict={}
        for i, exp in enumerate(exps):
            exp_dict[exp] = models[i]
        model_buffer.append(exp_dict)

    out= {}
    print (len(model_buffer))
    for model in model_buffer:
        for k,v in model.items():
            [out.setdefault(k, []).append(i) for i in v]

    applyTaylor(np.array(sat_buffer),np.array([v for k,v in out.items()]), years,exps)