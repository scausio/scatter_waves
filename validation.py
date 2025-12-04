"""
Wave Model Validation - Main Scatter Plot Module

This module provides comprehensive validation of wave model outputs (WAVEWATCH III)
against satellite observations. It generates publication-quality scatter plots with
detailed statistical metrics, supports percentile-based validation for extreme events,
and provides quantile analysis.

Features
--------
- Comprehensive statistical metrics (RMSE, BIAS, MAE, correlation, skill score, etc.)
- Percentile-based validation for extreme wave conditions
- Quantile-quantile (Q-Q) plots for distribution comparison
- Density-based scatter plots for large datasets
- Normalized and absolute error metrics
- Publication-quality graphics with detailed statistics tables

Author: Wave Model Validation Team
Date: 2024
License: MIT

Usage Example
-------------
python validation.py -c conf.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')
from stats import BIAS, RMSE, MAE, ScatterIndex, correlation, skill_score, symmetric_slope
import xarray as xr
import os
from scipy.stats import linregress, pearsonr, gaussian_kde
from utils import getConfigurationByID
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def calculate_comprehensive_stats(model_data: np.ndarray, 
                                  satellite_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistical metrics for model-observation comparison.
    
    Parameters
    ----------
    model_data : np.ndarray
        Model predictions (significant wave height)
    satellite_data : np.ndarray
        Satellite observations (significant wave height)
        
    Returns
    -------
    dict
        Dictionary containing all statistical metrics including:
        - Basic errors: bias, rmse, mae
        - Normalized errors: normalized_bias, normalized_rmse, normalized_mae
        - Correlation metrics: correlation, r2, p_value
        - Other metrics: scatter_index, skill_score, symmetric_slope
        - Data statistics: n_points, mean_obs, mean_model, std_obs, std_model
        
    Notes
    -----
    All NaN and negative values are automatically filtered out before computation.
    """
    # Data validation and cleaning
    valid_mask = (~np.isnan(model_data)) & (~np.isnan(satellite_data)) & \
                 (model_data >= 0) & (satellite_data >= 0)
    model_clean = np.array(model_data)[valid_mask]
    sat_clean = np.array(satellite_data)[valid_mask]
    
    if len(model_clean) == 0:
        raise ValueError("No valid data available after filtering")
    
    # Basic error metrics
    bias = BIAS(model_clean, sat_clean)
    rmse = RMSE(model_clean, sat_clean)
    mae = MAE(model_clean, sat_clean)
    
    # Correlation metrics
    corr = correlation(model_clean, sat_clean)
    r2 = r2_score(sat_clean, model_clean)
    
    # Statistical significance
    _, p_value = stats.pearsonr(model_clean, sat_clean)
    
    # Normalized metrics
    mean_obs = np.mean(sat_clean)
    normalized_bias = (bias / mean_obs * 100) if mean_obs > 0 else 0
    normalized_rmse = (rmse / mean_obs * 100) if mean_obs > 0 else 0
    normalized_mae = (mae / mean_obs * 100) if mean_obs > 0 else 0
    
    # Advanced metrics
    scatter_index = ScatterIndex(model_clean, sat_clean) * 100
    skill = skill_score(model_clean, sat_clean)
    sym_slope = symmetric_slope(model_clean, sat_clean)
    
    # Data statistics
    stats_dict = {
        'n_points': len(model_clean),
        'bias': bias,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr,
        'r2': r2,
        'p_value': p_value,
        'normalized_bias': normalized_bias,
        'normalized_rmse': normalized_rmse,
        'normalized_mae': normalized_mae,
        'scatter_index': scatter_index,
        'skill_score': skill,
        'symmetric_slope': sym_slope,
        'mean_obs': mean_obs,
        'mean_model': np.mean(model_clean),
        'std_obs': np.std(sat_clean),
        'std_model': np.std(model_clean),
        'median_obs': np.median(sat_clean),
        'median_model': np.median(model_clean),
        'q95_obs': np.percentile(sat_clean, 95),
        'q95_model': np.percentile(model_clean, 95)
    }
    
    return stats_dict


def create_qq_plot(ax: plt.Axes, model_data: np.ndarray, 
                  satellite_data: np.ndarray) -> None:
    """
    Create a Quantile-Quantile (Q-Q) plot to compare distributions.
    
    Q-Q plots help identify if the model and observations follow similar distributions.
    Points should fall along the 1:1 line for perfect agreement.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    model_data : np.ndarray
        Model predictions
    satellite_data : np.ndarray
        Satellite observations
    """
    # Calculate quantiles
    quantiles = np.linspace(0.01, 0.99, 100)
    model_quantiles = np.percentile(model_data, quantiles * 100)
    sat_quantiles = np.percentile(satellite_data, quantiles * 100)
    
    # Plot Q-Q
    ax.scatter(sat_quantiles, model_quantiles, alpha=0.6, s=20, color='#2E86AB')
    
    # 1:1 line
    min_val = min(np.min(sat_quantiles), np.min(model_quantiles))
    max_val = max(np.max(sat_quantiles), np.max(model_quantiles))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
            linewidth=2, label='1:1')
    
    ax.set_xlabel('Satellite Quantiles [m]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Model Quantiles [m]', fontsize=10, fontweight='bold')
    ax.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8)


def save_statistics_table(stats_data: list, title: str, save_path: str, 
                          dpi: int = 300, figsize: Tuple[int, int] = (12, 10)):
    """
    Save statistics table as a separate figure.
    
    This creates a dedicated figure containing only the statistics table,
    avoiding matplotlib's table rendering limitations in complex layouts.
    
    Parameters
    ----------
    stats_data : list
        Table data as list of lists, first row is header
    title : str
        Title for the table figure
    save_path : str
        Path to save the table figure
    dpi : int, optional
        Resolution for saved figure (default: 300)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (12, 10))
    """
    # Create dedicated figure for table
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Create table with generous spacing
    table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0],
                    cellLoc='left', loc='center',
                    colWidths=[0.35, 0.20, 0.45],
                    bbox=[0.05, 0.05, 0.90, 0.90])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Larger font since table has dedicated space
    table.scale(1.0, 2.5)  # Generous row height
    
    # Style cells
    for i in range(len(stats_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#2C3E50')
                cell.set_text_props(weight='bold', color='white', ha='left', fontsize=12)
                cell.set_height(0.05)
            else:
                # Check for separator or section headers
                if stats_data[i][0].startswith('---') or stats_data[i][0] == '':
                    cell.set_facecolor('#BDC3C7')
                    cell.set_text_props(weight='bold', ha='left', fontsize=11)
                    cell.set_height(0.04)
                elif i % 2 == 0:
                    cell.set_facecolor('#ECF0F1')
                    cell.set_height(0.035)
                else:
                    cell.set_facecolor('white')
                    cell.set_height(0.035)
            
            cell.set_edgecolor('#95A5A6')
            cell.set_linewidth(1.0)
            cell.PAD = 0.08
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    \u2713 Saved statistics table: {save_path}")


def generate_percentile_scatter(model_data: np.ndarray, satellite_data: np.ndarray,
                                percentile: int,
                                model_name: str = "Model", sat_name: str = "Satellite",
                                figsize: Tuple[int, int] = (14, 10), save_path: Optional[str] = None,
                                dpi: int = 300) -> plt.Figure:
    """
    Create scatter plot for data above a specific percentile threshold.
    
    This function generates a focused validation plot showing only extreme events
    (values above the specified percentile). Useful for validating model performance
    on high wave conditions.
    
    Parameters
    ----------
    model_data : array-like
        Model predictions (significant wave height in meters)
    satellite_data : array-like
        Satellite observations (significant wave height in meters)
    percentile : int
        Percentile threshold (e.g., 90 for 90th percentile)
        Only data >= this percentile will be plotted
    model_name : str, optional
        Name of model for labels (default: "Model")
    sat_name : str, optional
        Name of satellite/observation source (default: "Satellite")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (14, 10))
    save_path : str, optional
        Path to save figure. If None, figure is not saved (default: None)
    dpi : int, optional
        Resolution for saved figure (default: 300)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
        
    Raises
    ------
    ValueError
        If no valid data remains after filtering or if percentile results in < 10 points
        
    Examples
    --------
    >>> model = np.random.rand(1000) * 5 + 0.5
    >>> obs = model + np.random.randn(1000) * 0.3
    >>> fig = generate_percentile_scatter(model, obs, 
    ...                                   percentile=90,
    ...                                   model_name="WW3",
    ...                                   sat_name="Sentinel-3A",
    ...                                   save_path="validation_P90.png")
    
    Notes
    -----
    The percentile threshold is calculated from the satellite observations.
    Minimum of 10 points required for meaningful statistics.
    """
    # Data validation and cleaning
    valid_mask = (~np.isnan(model_data)) & (~np.isnan(satellite_data)) & \
                 (model_data >= 0) & (satellite_data >= 0)
    model_clean = np.array(model_data)[valid_mask]
    sat_clean = np.array(satellite_data)[valid_mask]
    
    if len(model_clean) == 0:
        raise ValueError("No valid data available after filtering")
    
    # Calculate percentile threshold from observations
    threshold = np.percentile(sat_clean, percentile)
    
    # Filter data above percentile
    perc_mask = sat_clean >= threshold
    model_perc = model_clean[perc_mask]
    sat_perc = sat_clean[perc_mask]
    
    if len(model_perc) < 10:
        raise ValueError(f"Insufficient data above P{percentile} ({len(model_perc)} points < 10)")
    
    print(f"    P{percentile} threshold: {threshold:.3f} m")
    print(f"    P{percentile} sample size: {len(model_perc)} points ({100*(1-percentile/100):.1f}% of data)")
    
    # Calculate statistics for percentile-filtered data
    stats_dict = calculate_comprehensive_stats(model_perc, sat_perc)
    
    # Set modern style
    sns.set_palette(sns.color_palette("viridis"))
    
    # Create figure WITHOUT table section (table will be separate)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 2, 0.8], 
                         width_ratios=[2, 0.8, 0.8],
                         hspace=0.3, wspace=0.4)
    
    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, 
                 f'{model_name} vs {sat_name} - P{percentile} Validation (Extreme Events ≥{threshold:.3f}m)', 
                 ha='center', va='center', fontsize=18, fontweight='bold',
                 color='#1a1a1a', transform=title_ax.transAxes)
    title_ax.axis('off')
    
    # ===== MAIN SCATTER PLOT =====
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Determine plot range
    max_val = max(np.max(sat_perc), np.max(model_perc))
    buffer = max_val * 0.15
    plot_min = threshold * 0.95  # Start slightly below threshold
    plot_max = max_val + buffer
    
    # Choose visualization method based on data size
    if len(model_perc) > 500:
        # Hexbin for large datasets
        scatter = ax_main.hexbin(sat_perc, model_perc, gridsize=80, 
                                cmap='plasma', mincnt=1, alpha=0.85, 
                                extent=[plot_min, plot_max, plot_min, plot_max],
                                edgecolors='face', linewidths=0.1)
    else:
        # Density-colored scatter for smaller datasets
        xy = np.vstack([sat_perc, model_perc])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = sat_perc[idx], model_perc[idx], z[idx]
        scatter = ax_main.scatter(x, y, c=z, alpha=0.75, cmap='plasma', 
                                 s=60, edgecolors='white', linewidth=0.3)
    
    # Colorbar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(scatter, cax=cax)
    cb.set_label('Density', fontsize=12, labelpad=10, fontweight='bold')

    # 1:1 line (perfect agreement)
    ax_main.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
                linewidth=2.5, label='1:1', alpha=0.9, zorder=5)
    
    # Regression line
    slope, intercept, r_value, p_val, std_err = stats.linregress(sat_perc, model_perc)
    line_x = np.array([plot_min, plot_max])
    line_y = slope * line_x + intercept
    ax_main.plot(line_x, line_y, color='#FF6B35', linewidth=2.5, alpha=0.9,
                label=f'Regression (y={slope:.2f}x+{intercept:.2f})', zorder=5)
    
    # Confidence bands (±RMSE)
    rmse_val = stats_dict['rmse']
    ax_main.fill_between([plot_min, plot_max], 
                        [plot_min - rmse_val, plot_max - rmse_val],
                        [plot_min + rmse_val, plot_max + rmse_val], 
                        alpha=0.15, color='orange', label=f'±RMSE ({rmse_val:.3f}m)', zorder=1)
    
    # Percentile threshold line
    ax_main.axvline(x=threshold, color='green', linestyle=':', linewidth=2, 
                   label=f'P{percentile} threshold ({threshold:.3f}m)', alpha=0.7)
    ax_main.axhline(y=threshold, color='green', linestyle=':', linewidth=2, alpha=0.7)
    
    # Styling
    ax_main.set_xlabel(f'{sat_name} SWH [m]', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f'{model_name} SWH [m]', fontsize=14, fontweight='bold')
    ax_main.set_xlim(plot_min, plot_max)
    ax_main.set_ylim(plot_min, plot_max)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
    ax_main.legend(loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, fontsize=9, framealpha=0.95)
    ax_main.set_aspect('equal')
    
    # Calculate max density for shared Y-axis
    kde_sat = gaussian_kde(sat_perc)
    kde_model = gaussian_kde(model_perc)
    x_range = np.linspace(plot_min, plot_max, 200)
    max_density = max(np.max(kde_sat(x_range)), np.max(kde_model(x_range))) * 1.1
    
    # ===== MARGINAL DISTRIBUTION - SATELLITE (TOP) =====
    ax_top = fig.add_subplot(gs[1, 1])
    # Histogram goes UP (vertical bars)
    ax_top.hist(sat_perc, bins=30, alpha=0.7, color='#4ECDC4',
                edgecolor='black', linewidth=0.5, density=True, orientation='vertical')
    
    # Add KDE overlay - curve goes UP
    ax_top.plot(x_range, kde_sat(x_range), 'k-', linewidth=2)
    
    ax_top.set_xlim(plot_min, plot_max)
    ax_top.set_ylim(0, max_density)  # Shared Y-axis scale
    ax_top.set_title(f'{sat_name}\\nDistribution', fontsize=10, fontweight='bold')
    ax_top.set_ylabel('Density', fontsize=8)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.tick_params(labelsize=8)
    
    # ===== MARGINAL DISTRIBUTION - MODEL (RIGHT) =====
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_top)  # Share Y-axis with top
    # Histogram goes UP (vertical bars) - SAME orientation as top
    ax_right.hist(model_perc, bins=30, alpha=0.7, color='#F38181',
                 edgecolor='black', linewidth=0.5, density=True, orientation='vertical')
    
    # Add KDE overlay - curve goes UP
    ax_right.plot(x_range, kde_model(x_range), 'k-', linewidth=2)
    
    ax_right.set_xlim(plot_min, plot_max)
    ax_right.set_ylim(0, max_density)  # Shared Y-axis scale (same as top)
    ax_right.set_title(f'{model_name}\\nDistribution', fontsize=10, fontweight='bold')
    ax_right.set_ylabel('Density', fontsize=8)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.tick_params(labelsize=8)
    
    # ===== BUILD STATISTICS TABLE DATA (will be saved separately) =====
    stats_data = [['Metric', 'Value', 'Description']]
    
    # Sample information
    stats_data.extend([
        ['--- Sample Information ---', '', ''],
        ['N (Total)', f"{len(model_clean)}", 'Total valid observations'],
        [f'N (P{percentile})', f"{stats_dict['n_points']}", f'Observations ≥ P{percentile}'],
        [f'P{percentile} Threshold (m)', f"{threshold:.4f}", f'{percentile}th percentile of observations'],
        [f'% Above P{percentile}', f"{100*(1-percentile/100):.1f}%", 'Percentage in extreme range'],
    ])
    
    # Correlation metrics
    stats_data.extend([
        ['', '', ''],
        ['--- Correlation Metrics ---', '', ''],
        ['Pearson ρ', f"{stats_dict['correlation']:.2f}", 'Correlation coefficient'],
        ['R²', f"{stats_dict['r2']:.2f}", 'Coefficient of determination'],
        ['p-value', f"{stats_dict['p_value']:.3e}", 'Statistical significance'],
    ])
    
    # Error metrics
    stats_data.extend([
        ['', '', ''],
        ['--- Error Metrics ---', '', ''],
        ['BIAS (m)', f"{stats_dict['bias']:.2f}", 'Mean systematic error'],
        ['RMSE (m)', f"{stats_dict['rmse']:.2f}", 'Root Mean Square Error'],
        ['MAE (m)', f"{stats_dict['mae']:.2f}", 'Mean Absolute Error'],
        ['NBIAS (%)', f"{stats_dict['normalized_bias']:.2f}", 'Normalized bias'],
        ['NRMSE (%)', f"{stats_dict['normalized_rmse']:.2f}", 'Normalized RMSE'],
        ['SI (%)', f"{stats_dict['scatter_index']:.2f}", 'Scatter Index'],
    ])
    
    # # Advanced metrics
    # stats_data.extend([
    #     ['', '', ''],
    #     ['--- Advanced Metrics ---', '', ''],
    #     ['Skill Score', f"{stats_dict['skill_score']:.4f}", 'Murphy skill score'],
    #     ['Symmetric Slope', f"{stats_dict['symmetric_slope']:.4f}", 'Best fit'],
    # ])
    
    # Data statistics
    stats_data.extend([
        ['', '', ''],
        ['--- Extreme Event Statistics ---', '', ''],
        ['Mean Obs (m)', f"{stats_dict['mean_obs']:.2f}", 'Mean extreme wave height (obs)'],
        ['Mean Model (m)', f"{stats_dict['mean_model']:.2f}", 'Mean extreme wave height (model)'],
        ['Std Obs (m)', f"{stats_dict['std_obs']:.2f}", 'Std deviation (obs)'],
        ['Std Model (m)', f"{stats_dict['std_model']:.2f}", 'Std deviation (model)'],
        ['Max Obs (m)', f"{np.max(sat_perc):.2f}", 'Maximum observed value'],
        ['Max Model (m)', f"{np.max(model_perc):.2f}", 'Maximum modeled value'],
    ])
    
    plt.tight_layout()
    
    # Save scatter figure (without table)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"    \u2713 Saved P{percentile} scatter plot: {save_path}")
        
        # Save statistics table as separate figure
        # Generate table filename based on scatter filename
        table_path = save_path.replace('.jpeg', '_stats_table.jpeg').replace('.jpg', '_stats_table.jpg').replace('.png', '_stats_table.png')
        table_title = f'Statistical Metrics: P{percentile} Validation (Extreme Events ≥{threshold:.3f}m)\n{model_name} vs {sat_name}'
        save_statistics_table(stats_data, table_title, table_path, dpi=dpi)
    
    return fig


def scatter_waves(model_data: np.ndarray, satellite_data: np.ndarray,
                 model_name: str = "Model", sat_name: str = "Satellite",
                 figsize: Tuple[int, int] = (16, 10), save_path: Optional[str] = None,
                 dpi: int = 300, percentile_thresholds: Optional[List[int]] = None,
                 include_qq_plot: bool = True,
                 generate_percentile_plots: bool = True) -> plt.Figure:
    """
    Create comprehensive validation scatter plot with advanced statistical analysis.
    
    This function generates a multi-panel figure including:
    1. Main scatter plot with density coloring
    2. Marginal distribution histograms
    3. Q-Q plot for distribution comparison
    4. Detailed statistics table
    5. Percentile-based validation (optional)
    
    Parameters
    ----------
    model_data : array-like
        Model predictions (significant wave height in meters)
    satellite_data : array-like
        Satellite observations (significant wave height in meters)
    model_name : str, optional
        Name of model for labels (default: "Model")
    sat_name : str, optional
        Name of satellite/observation source (default: "Satellite")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (16, 10))
    save_path : str, optional
        Path to save figure. If None, figure is not saved (default: None)
    dpi : int, optional
        Resolution for saved figure (default: 300)
    percentile_thresholds : list of int, optional
        Percentiles for threshold-based validation (e.g., [75, 90, 95])
        If provided, additional statistics for extreme values are computed
    include_qq_plot : bool, optional
        Whether to include Q-Q plot (default: True)
    generate_percentile_plots : bool, optional
        Whether to generate separate scatter plots for each percentile threshold (default: True)
        Creates additional plots showing only data above each percentile
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
        
    Raises
    ------
    ValueError
        If no valid data remains after filtering NaN and negative values
        
    Examples
    --------
    >>> model = np.random.rand(1000) * 5 + 0.5
    >>> obs = model + np.random.randn(1000) * 0.3
    >>> fig = scatter_waves(model, obs, 
    ...                     model_name="WW3",
    ...                     sat_name="Sentinel-3A",
    ...                     percentile_thresholds=[90, 95],
    ...                     save_path="validation_plot.png")
    
    Notes
    -----
    For large datasets (>1000 points), hexbin plots are used for better visualization.
    For smaller datasets, kernel density estimation coloring is applied to scatter points.
    """
    # Data validation and cleaning
    valid_mask = (~np.isnan(model_data)) & (~np.isnan(satellite_data)) & \
                 (model_data >= 0) & (satellite_data >= 0)
    model_clean = np.array(model_data)[valid_mask]
    sat_clean = np.array(satellite_data)[valid_mask]
    
    if len(model_clean) == 0:
        raise ValueError("No valid data available after filtering")
    
    # Calculate comprehensive statistics
    stats_dict = calculate_comprehensive_stats(model_clean, sat_clean)
    
    # Set modern style
    sns.set_palette(sns.color_palette("viridis"))
    
    # Create figure with enhanced layout
    n_cols = 4 if include_qq_plot else 3
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Define grid layout WITHOUT table section (table will be saved separately)
    if include_qq_plot:
        gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 2, 0.8], 
                            width_ratios=[2, 0.8, 0.8, 1.2],
                            hspace=0.3, wspace=0.4)
    else:
        gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 2, 0.8], 
                            width_ratios=[2, 0.8, 0.8],
                            hspace=0.3, wspace=0.4)
    
    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, f'{model_name} vs {sat_name} - Comprehensive Validation', 
                 ha='center', va='center', fontsize=20, fontweight='bold',
                 color='#1a1a1a', transform=title_ax.transAxes)
    title_ax.axis('off')
    
    # ===== MAIN SCATTER PLOT =====
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Determine plot range
    max_val = max(np.max(sat_clean), np.max(model_clean))
    buffer = max_val * 0.15
    plot_min = 0
    plot_max = max_val + buffer
    
    # Choose visualization method based on data size
    if len(model_clean) > 1000:
        # Hexbin for large datasets
        scatter = ax_main.hexbin(sat_clean, model_clean, gridsize=120, 
                                cmap='viridis', mincnt=1, alpha=0.85, 
                                extent=[plot_min, plot_max, plot_min, plot_max],
                                edgecolors='face', linewidths=0.1)
    else:
        # Density-colored scatter for smaller datasets
        xy = np.vstack([sat_clean, model_clean])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = sat_clean[idx], model_clean[idx], z[idx]
        scatter = ax_main.scatter(x, y, c=z, alpha=0.75, cmap='viridis', 
                                 s=50, edgecolors='white', linewidth=0.3)
    
    # Colorbar
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(scatter, cax=cax)
    cb.set_label('Density', fontsize=12, labelpad=10, fontweight='bold')
    
    # 1:1 line (perfect agreement)
    ax_main.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
                linewidth=2.5, label='1:1', alpha=0.9, zorder=5)
    
    # Regression line
    slope, intercept, r_value, p_val, std_err = stats.linregress(sat_clean, model_clean)
    line_x = np.array([plot_min, plot_max])
    line_y = slope * line_x + intercept
    ax_main.plot(line_x, line_y, color='#FF6B35', linewidth=2.5, alpha=0.9,
                label=f'Regression (y={slope:.2f}x+{intercept:.2f})', zorder=5)
    
    # Confidence bands (±RMSE)
    rmse_val = stats_dict['rmse']
    ax_main.fill_between([plot_min, plot_max], 
                        [plot_min - rmse_val, plot_max - rmse_val],
                        [plot_min + rmse_val, plot_max + rmse_val], 
                        alpha=0.15, color='orange', label=f'±RMSE ({rmse_val:.3f}m)', zorder=1)
    
    # Styling
    ax_main.set_xlabel(f'{sat_name} SWH [m]', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f'{model_name} SWH [m]', fontsize=14, fontweight='bold')
    ax_main.set_xlim(plot_min, plot_max)
    ax_main.set_ylim(plot_min, plot_max)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
    ax_main.legend(loc='upper left', frameon=True, fancybox=True, 
                  shadow=True, fontsize=10, framealpha=0.95)
    ax_main.set_aspect('equal')
    
    # Calculate max density for shared Y-axis
    kde_sat = gaussian_kde(sat_clean)
    kde_model = gaussian_kde(model_clean)
    x_range = np.linspace(plot_min, plot_max, 200)
    max_density = max(np.max(kde_sat(x_range)), np.max(kde_model(x_range))) * 1.1
    
    # ===== MARGINAL DISTRIBUTION - SATELLITE (TOP) =====
    ax_top = fig.add_subplot(gs[1, 1])
    # Histogram goes UP (vertical bars)
    ax_top.hist(sat_clean, bins=40, alpha=0.7, color='#4ECDC4',
                edgecolor='black', linewidth=0.5, density=True, orientation='vertical')
    
    # Add KDE overlay - curve goes UP
    ax_top.plot(x_range, kde_sat(x_range), 'k-', linewidth=2, label='KDE')
    
    ax_top.set_xlim(plot_min, plot_max)
    ax_top.set_ylim(0, max_density)  # Shared Y-axis scale
    ax_top.set_title(f'{sat_name}\nDistribution', fontsize=11, fontweight='bold')
    ax_top.set_ylabel('Density', fontsize=9)
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.legend(fontsize=8)
    
    # ===== MARGINAL DISTRIBUTION - MODEL (RIGHT) =====
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_top)  # Share Y-axis with top
    # Histogram goes UP (vertical bars) - SAME orientation as top
    ax_right.hist(model_clean, bins=40, alpha=0.7, color='#F38181',
                 edgecolor='black', linewidth=0.5, density=True, orientation='vertical')
    
    # Add KDE overlay - curve goes UP
    ax_right.plot(x_range, kde_model(x_range), 'k-', linewidth=2, label='KDE')
    
    ax_right.set_xlim(plot_min, plot_max)
    ax_right.set_ylim(0, max_density)  # Shared Y-axis scale (same as top)
    ax_right.set_title(f'{model_name}\nDistribution', fontsize=11, fontweight='bold')
    ax_right.set_ylabel('Density', fontsize=9)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.legend(fontsize=8)
    
    # ===== Q-Q PLOT (if requested) =====
    if include_qq_plot:
        ax_qq = fig.add_subplot(gs[1, 3])
        create_qq_plot(ax_qq, model_clean, sat_clean)
    
    # ===== BUILD STATISTICS TABLE DATA (will be saved separately) =====
    # Prepare statistics table data
    stats_data = [
        ['Metric', 'Value', 'Description']
    ]
    
    # Basic statistics
    stats_data.extend([
        ['N (observations)', f"{stats_dict['n_points']:,}", 'Number of valid data points'],
        ['Pearson ρ', f"{stats_dict['correlation']:.4f}", 'Correlation coefficient (-1 to 1)'],
        ['R²', f"{stats_dict['r2']:.4f}", 'Coefficient of determination (0 to 1)'],
        ['p-value', f"{stats_dict['p_value']:.4e}", 'Statistical significance of correlation'],
    ])
    
    # Error metrics
    stats_data.extend([
        ['', '', ''],  # Separator
        ['--- Absolute Errors ---', '', ''],
        ['BIAS (m)', f"{stats_dict['bias']:.2f}", 'Mean systematic error (model - obs)'],
        ['RMSE (m)', f"{stats_dict['rmse']:.2f}", 'Root Mean Square Error'],
        ['MAE (m)', f"{stats_dict['mae']:.2f}", 'Mean Absolute Error'],
    ])
    
    # Normalized metrics
    stats_data.extend([
        ['', '', ''],  # Separator
        ['--- Normalized Errors ---', '', ''],
        ['NBIAS (%)', f"{stats_dict['normalized_bias']:.2f}", 'Normalized bias'],
        ['NRMSE (%)', f"{stats_dict['normalized_rmse']:.2f}", 'Normalized RMSE'],
        ['NMAE (%)', f"{stats_dict['normalized_mae']:.2f}", 'Normalized MAE'],
        ['SI (%)', f"{stats_dict['scatter_index']:.2f}", 'Scatter Index'],
    ])
    
    # # Advanced metrics
    # stats_data.extend([
    #     ['', '', ''],  # Separator
    #     ['--- Advanced Metrics ---', '', ''],
    #     ['Skill Score', f"{stats_dict['skill_score']:.4f}", 'Murphy skill score (>0 is good)'],
    #     ['Symmetric Slope', f"{stats_dict['symmetric_slope']:.4f}", 'Best fit'],
    # ])
    
    # Data statistics
    stats_data.extend([
        ['', '', ''],  # Separator
        ['--- Data Statistics ---', '', ''],
        ['Mean Obs (m)', f"{stats_dict['mean_obs']:.2f}", 'Mean observed value'],
        ['Mean Model (m)', f"{stats_dict['mean_model']:.2f}", 'Mean modeled value'],
        ['Std Obs (m)', f"{stats_dict['std_obs']:.2f}", 'Observation standard deviation'],
        ['Std Model (m)', f"{stats_dict['std_model']:.2f}", 'Model standard deviation'],
        ['Median Obs (m)', f"{stats_dict['median_obs']:.2f}", 'Median observed value'],
        ['Median Model (m)', f"{stats_dict['median_model']:.2f}", 'Median modeled value'],
        ['95th %ile Obs (m)', f"{stats_dict['q95_obs']:.2f}", '95th percentile observed'],
        ['95th %ile Model (m)', f"{stats_dict['q95_model']:.2f}", '95th percentile modeled'],
    ])
    
    # Percentile-based validation (for extreme events)
    if percentile_thresholds:
        stats_data.extend([
            ['', '', ''],  # Separator
            ['--- Percentile-Based Validation ---', '', '']
        ])
        
        for perc in percentile_thresholds:
            threshold = np.percentile(sat_clean, perc)
            mask = sat_clean >= threshold
            
            if np.sum(mask) > 1:
                model_high = model_clean[mask]
                obs_high = sat_clean[mask]
                
                bias_p = BIAS(model_high, obs_high)
                rmse_p = RMSE(model_high, obs_high)
                mae_p = MAE(model_high, obs_high)
                corr_p = correlation(model_high, obs_high)
                
                stats_data.extend([
                    [f'P{perc} Threshold (m)', f'{threshold:.3f}', f'Values ≥ {perc}th percentile'],
                    [f'P{perc} N', f'{np.sum(mask)}', f'Count above {perc}th percentile'],
                    [f'P{perc} BIAS (m)', f'{bias_p:.4f}', f'Bias for extreme values (≥P{perc})'],
                    [f'P{perc} RMSE (m)', f'{rmse_p:.4f}', f'RMSE for extreme values (≥P{perc})'],
                    [f'P{perc} MAE (m)', f'{mae_p:.4f}', f'MAE for extreme values (≥P{perc})'],
                    [f'P{perc} Correlation', f'{corr_p:.4f}', f'Correlation for extreme values (≥P{perc})'],
                ])
    
    # Footer (kept for visual branding on scatter plot)
    fig.text(0.99, 0.01, f'Generated: {model_name} Validation', 
            ha='right', va='bottom', alpha=0.4, fontsize=8, style='italic')
    
    # Save scatter figure (without table)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Validation plot saved: {save_path}")
        print(f"  - Data points: {stats_dict['n_points']:,}")
        print(f"  - Correlation: {stats_dict['correlation']:.3f}")
        print(f"  - RMSE: {stats_dict['rmse']:.3f} m")
        print(f"  - Bias: {stats_dict['bias']:.3f} m")
        
        # Save statistics table as separate figure
        # Generate table filename based on scatter filename
        table_path = save_path.replace('.jpeg', '_stats_table.jpeg').replace('.jpg', '_stats_table.jpg').replace('.png', '_stats_table.png')
        table_title = f'Comprehensive Statistical Metrics: {model_name} vs {sat_name}'
        save_statistics_table(stats_data, table_title, table_path, dpi=dpi)
    
    return fig


def simple_scatter_plot(model_data: np.ndarray, satellite_data: np.ndarray,
                        model_name: str = "Model", sat_name: str = "Satellite",
                        figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = None,
                        dpi: int = 300) -> plt.Figure:
    """
    Create a simple, clean scatter plot with only the main scatter and basic statistics.
    
    This function generates a minimalist scatter plot without marginal distributions,
    Q-Q plots, or detailed statistics tables. Ideal for quick visual assessment,
    presentations where simplicity is preferred, or when space is limited.
    
    Parameters
    ----------
    model_data : array-like
        Model predictions (significant wave height in meters)
    satellite_data : array-like
        Satellite observations (significant wave height in meters)
    model_name : str, optional
        Name of model for labels (default: "Model")
    sat_name : str, optional
        Name of satellite/observation source (default: "Satellite")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 10))
    save_path : str, optional
        Path to save figure. If None, figure is not saved (default: None)
    dpi : int, optional
        Resolution for saved figure (default: 300)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
        
    Raises
    ------
    ValueError
        If no valid data remains after filtering NaN and negative values
        
    Examples
    --------
    >>> model = np.random.rand(1000) * 5 + 0.5
    >>> obs = model + np.random.randn(1000) * 0.3
    >>> fig = simple_scatter_plot(model, obs, 
    ...                           model_name="WW3",
    ...                           sat_name="Sentinel-3A",
    ...                           save_path="simple_scatter.png")
    
    Notes
    -----
    This function provides a cleaner, simpler alternative to the comprehensive
    scatter_waves() function. It shows only:
    - Main scatter plot with density coloring
    - 1:1 reference line
    - Regression line
    - Basic statistics overlay (correlation, RMSE, bias, N)
    """
    # Data validation and cleaning
    valid_mask = (~np.isnan(model_data)) & (~np.isnan(satellite_data)) & \
                 (model_data >= 0) & (satellite_data >= 0)
    model_clean = np.array(model_data)[valid_mask]
    sat_clean = np.array(satellite_data)[valid_mask]
    
    if len(model_clean) == 0:
        raise ValueError("No valid data available after filtering")
    
    # Calculate basic statistics
    bias = BIAS(model_clean, sat_clean)
    rmse = RMSE(model_clean, sat_clean)
    mae = MAE(model_clean, sat_clean)
    corr = correlation(model_clean, sat_clean)
    n_points = len(model_clean)
    
    # Set modern style
    sns.set_palette(sns.color_palette("viridis"))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Determine plot range
    max_val = max(np.max(sat_clean), np.max(model_clean))
    buffer = max_val * 0.15
    plot_min = 0
    plot_max = max_val + buffer
    
    # Choose visualization method based on data size
    if len(model_clean) > 1000:
        # Hexbin for large datasets
        scatter = ax.hexbin(sat_clean, model_clean, gridsize=120, 
                           cmap='viridis', mincnt=1, alpha=0.85, 
                           extent=[plot_min, plot_max, plot_min, plot_max],
                           edgecolors='face', linewidths=0.1)
    else:
        # Density-colored scatter for smaller datasets
        xy = np.vstack([sat_clean, model_clean])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = sat_clean[idx], model_clean[idx], z[idx]
        scatter = ax.scatter(x, y, c=z, alpha=0.75, cmap='viridis', 
                            s=60, edgecolors='white', linewidth=0.3)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(scatter, cax=cax)
    cb.set_label('Density', fontsize=12, labelpad=10, fontweight='bold')
    # 1:1 line (perfect agreement)
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', 
           linewidth=2.5, label='1:1', alpha=0.9, zorder=5)
    
    # Regression line
    slope, intercept, r_value, p_val, std_err = stats.linregress(sat_clean, model_clean)
    line_x = np.array([plot_min, plot_max])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='#FF6B35', linewidth=2.5, alpha=0.9,
           label=f'y={slope:.2f}x+{intercept:.2f}', zorder=5)
    
    # Styling
    ax.set_xlabel(f'{sat_name}', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{model_name}', fontsize=16, fontweight='bold')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.7)
    ax.legend(loc='upper left', frameon=True, fancybox=True, 
             shadow=True, fontsize=12, framealpha=0.95)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=12)
    
    # Title
    ax.set_title(f'{model_name} vs {sat_name} - Validation Scatter Plot', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add statistics text box
    stats_text = f'N = {n_points:,}\n'
    stats_text += f'ρ = {corr:.2f}\n'
    stats_text += f'RMSE = {rmse:.2f} m\n'
    stats_text += f'BIAS = {bias:.2f} m\n'
    stats_text += f'MAE = {mae:.2f} m'
    
    # Position text box in upper right
    ax.text(0.97, 0.03, stats_text,
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='#2C3E50', linewidth=2),
           fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Simple scatter plot saved: {save_path}")
    
    return fig


def maskNtimes(model: np.ndarray, sat: np.ndarray, times: float) -> np.ndarray:
    """
    Create mask for outliers where difference exceeds threshold.
    
    This filter identifies data points where the absolute difference between
    model and satellite exceeds a multiple of the model value. Useful for
    removing physically unrealistic outliers.
    
    Parameters
    ----------
    model : np.ndarray
        Model predictions
    sat : np.ndarray
        Satellite observations
    times : float
        Multiplier threshold (e.g., 3.0 means mask if |diff| > 3*model)
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = outlier to be removed)
        
    Examples
    --------
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> sat = np.array([1.1, 8.0, 2.9])
    >>> mask = maskNtimes(model, sat, 2.0)
    >>> print(mask)  # [False, True, False] - middle point is outlier
    """
    diff = np.abs(model - sat)
    print(f'Max difference: {np.nanmax(diff):.3f} m')
    print(f'Model * threshold: {np.nanmax(model * times):.3f} m')
    print(f'Threshold value: {times}')
    return diff > (model * times)


def main(conf_path: str, start_date: str, end_date: str) -> None:
    """
    Main validation workflow for wave model outputs.
    
    This function:
    1. Loads configuration from YAML file
    2. Reads model and satellite data
    3. Applies quality filters (thresholds, ntimes filter)
    4. Generates comprehensive validation plots with statistics
    5. Optionally performs percentile-based validation
    
    Parameters
    ----------
    conf_path : str
        Path to configuration YAML file
    start_date : str
        Start date for validation period (YYYYMMDD format)
    end_date : str
        End date for validation period (YYYYMMDD format)
        
    Notes
    -----
    Configuration file must contain 'plot' section with:
    - experiments: dict of model experiments to validate
    - out_dir: output directory for plots
    - filters: quality control parameters (min, max, ntimes, unbias)
    - title: plot title
    
    Optional configuration:
    - percentile_thresholds: list of percentiles for extreme value analysis
      Example: [75, 90, 95]
    
    Examples
    --------
    >>> main('conf.yaml', '20230901', '20230930')
    """
    # Load configuration
    conf = getConfigurationByID(conf_path, 'plot')
    outdir = os.path.join(conf.out_dir.out_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)
    date = f"{start_date}_{end_date}"
    
    print("="*80)
    print(f"WAVE MODEL VALIDATION: {start_date} to {end_date}")
    print("="*80)
    
    # Get percentile thresholds from configuration (if provided)
    percentile_thresholds = getattr(conf.filters, 'percentile_thresholds', [90, 95])
    dpi = 300  # Default DPI for high-quality output
    
    print(f"\nConfiguration:")
    print(f"  Percentile thresholds: {percentile_thresholds}")
    print(f"  Output DPI: {dpi}")
    
    # Data containers
    ds = {}
    
    # Load and process each experiment
    for i, dataset in enumerate(conf.experiments):
        print(f"\nProcessing experiment: {dataset}")
        
        # Load dataset
        ds_path = (conf.experiments[dataset].series).format(
            out_dir=conf.out_dir.out_dir, date=date)
        ds_all = xr.open_dataset(ds_path)
        
        print(f"  Dataset shape: {ds_all.dims}")
        
        # Extract variables
        sat_hs = ds_all.hs
        model_hs = ds_all.model_hs
        
        # Remove rows with any NaN in model
        model_hs = model_hs.where(~np.any(np.isnan(model_hs), axis=1), np.nan)
        
        print(f"  Satellite NaN count: {np.sum(np.isnan(sat_hs))}")
        print(f"  Satellite range: [{np.nanmin(sat_hs.values):.3f}, {np.nanmax(sat_hs.values):.3f}] m")
        
        # Apply threshold filters
        sat_hs = sat_hs.where(
            (ds_all.hs.values <= float(conf.filters.max)) & 
            (ds_all.hs.values >= float(conf.filters.min)))
        model_hs = model_hs.where(
            (ds_all.model_hs.values <= float(conf.filters.max)) & 
            (ds_all.model_hs.values >= float(conf.filters.min)))
        
        # Store data
        ds['sat'] = sat_hs.values
        ds[dataset] = model_hs.sel(model=dataset).values
        
        # Initialize validity mask on first iteration
        if i == 0:
            notValid = np.isnan(ds['sat'])
        
        print(f"  Data pairs: {len(ds['sat'])}")
        
        # Update validity mask
        notValid = notValid | np.isnan(ds[dataset])
        
        print(f"  Valid pairs after NaN filter: {np.sum(~notValid)}")
        
        # Apply ntimes filter if configured
        if hasattr(conf.filters, 'ntimes') and conf.filters.ntimes:
            print(f"  Applying ntimes filter (threshold: {conf.filters.ntimes})")
            ntimes = maskNtimes(ds[dataset], ds['sat'], float(conf.filters.ntimes))
            notValid = notValid | ntimes
            print(f"  Valid pairs after ntimes filter: {np.sum(~notValid)}")
    
    # Extract valid data for plotting
    sat2plot = ds['sat'][np.argwhere(~notValid)[:, 0]]
    
    print(f"\n{'='*80}")
    print(f"GENERATING VALIDATION PLOTS")
    print(f"{'='*80}")
    
    # Generate plots for each experiment
    for i, dataset in enumerate(conf.experiments):
        print(f"\nExperiment: {dataset}")
        
        outName = os.path.join(outdir, f'scatter_{dataset}_{date}.jpeg')
        mod2plot = ds[dataset][np.argwhere(~notValid)[:, 0]]
        
        # Apply unbiasing if configured (only on first iteration)
        if hasattr(conf.filters, 'unbias') and \
           (conf.filters.unbias in ['True', 'T', 'TRUE', 't']) and (i == 0):
            print("  Applying unbias correction")
            sat2plot -= np.nanmean(sat2plot)
            sat2plot += np.nanmean(mod2plot)
        
        # Generate comprehensive scatter plot
        try:
            fig = scatter_waves(
                mod2plot, sat2plot,
                model_name=f"{dataset} SWH [m]",
                sat_name="Satellite SWH [m]",
                save_path=outName,
                percentile_thresholds=percentile_thresholds,
                include_qq_plot=True,
                generate_percentile_plots=True)
            plt.close(fig)
            print(f"  ✓ Generated: {os.path.basename(outName)}")
            
            # Generate simple scatter plot (scatter only, no tables/distributions)
            print(f"  Generating simple scatter plot...")
            try:
                simple_outName = outName.replace(f'scatter_{dataset}', f'scatter_{dataset}_simple')
                fig_simple = simple_scatter_plot(
                    mod2plot, sat2plot,
                    model_name=f"{dataset} SWH [m]",
                    sat_name="Satellite SWH [m]",
                    save_path=simple_outName,
                    dpi=dpi)
                plt.close(fig_simple)
                print(f"  ✓ Generated simple scatter plot")
            except Exception as e_simple:
                print(f"  ✗ Error generating simple plot: {e_simple}")
            
            # Generate separate scatter plots for percentile thresholds
            if percentile_thresholds and len(percentile_thresholds) > 0:
                for perc in percentile_thresholds:
                    print(f"  Generating P{perc} scatter plot...")
                    try:
                        fig_perc = generate_percentile_scatter(
                            mod2plot, sat2plot,
                            percentile=perc,
                            model_name=f"{dataset} SWH [m]",
                            sat_name="Satellite SWH [m]",
                            save_path=outName.replace(f'scatter_{dataset}', f'scatter_{dataset}_P{perc}'),
                            dpi=dpi)
                        plt.close(fig_perc)
                        print(f"    ✓ Generated P{perc} scatter plot")
                    except Exception as e_perc:
                        print(f"    ✗ Error generating P{perc} plot: {e_perc}")
                        
        except Exception as e:
            print(f"  ✗ Error generating plot: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"Output directory: {outdir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Wave Model Validation - Scatter Plot Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validation.py -c conf.yaml
  
For more information, see README.md
        """)
    
    parser.add_argument('-c', '--config', required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--start-date', 
                       help='Start date (YYYYMMDD) - overrides config')
    parser.add_argument('--end-date',
                       help='End date (YYYYMMDD) - overrides config')
    
    args = parser.parse_args()
    
    # For now, this requires dates to be passed from main.py
    # This is just for standalone usage documentation
    print("Note: This script is typically called from main.py")
