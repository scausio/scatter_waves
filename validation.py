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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stats import BIAS, RMSE, MAE, ScatterIndex, correlation, skill_score, symmetric_slope, summary
import os
from scipy.stats import gaussian_kde
from pairs import load_pairs, load_plot_conf, plot_dir
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
        - Correlation metrics: correlation, r2 (squared correlation), p_value
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
    r2 = corr ** 2
    
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
                                figsize: Tuple[int, int] = (14, 7), save_path: Optional[str] = None,
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
    # Create figure WITHOUT table section (table will be separate)
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 2], 
                         width_ratios=[2, 0.8, 0.8],
                         hspace=0.3, wspace=0.55)
    
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
    ax_main.set_xlabel(f'{sat_name}', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f'{model_name}', fontsize=14, fontweight='bold')
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
    ax_right.tick_params(labelleft=False)
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
        ['R²', f"{stats_dict['r2']:.2f}", 'Squared correlation'],
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
                 figsize: Tuple[int, int] = (18, 7), save_path: Optional[str] = None,
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
    # Create figure with enhanced layout
    n_cols = 4 if include_qq_plot else 3
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Define grid layout WITHOUT table section (table will be saved separately)
    if include_qq_plot:
        gs = fig.add_gridspec(2, 4, height_ratios=[0.1, 2], 
                            width_ratios=[2, 0.8, 0.8, 1.2],
                            hspace=0.3, wspace=0.55)
    else:
        gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 2], 
                            width_ratios=[2, 0.8, 0.8],
                            hspace=0.3, wspace=0.55)
    
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
    ax_main.set_xlabel(f'{sat_name}', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f'{model_name}', fontsize=14, fontweight='bold')
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
    ax_right.tick_params(labelleft=False)
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
        ['R²', f"{stats_dict['r2']:.4f}", 'Squared correlation (0 to 1)'],
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
    
    stats_data.extend([
        ['', '', ''],  # Separator
        ['--- Advanced Metrics ---', '', ''],
        ['Skill Score', f"{stats_dict['skill_score']:.4f}", 'Murphy skill score 1-MSE/var(obs) (>0 is good)'],
        ['Symmetric Slope', f"{stats_dict['symmetric_slope']:.4f}", 'Mean of direct and inverse regression slopes'],
    ])
    
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


COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
MARKERS = ['s', '^', 'v', 'D', 'P', 'X', '<', '>', 'h', '*']


def metrics_table(pairs) -> pd.DataFrame:
    """Summary metrics for each model, overall and per satellite mission."""
    rows = []
    satellites = ['ALL'] + sorted(set(pairs.satellite.values) - {'ALL'})
    for model in pairs.model.values:
        mod = pairs.model_hs.sel(model=model).values
        for sat in satellites:
            sel = slice(None) if sat == 'ALL' else pairs.satellite.values == sat
            rows.append({'model': model, 'satellite': sat, **summary(mod[sel], pairs.hs.values[sel])})
    return pd.DataFrame(rows).set_index(['model', 'satellite'])


def taylor_diagram(table: pd.DataFrame, save_path: str, title: str = '') -> None:
    """Taylor diagram: normalised standard deviation vs correlation.

    Large markers: each model on all pairs; small markers: per satellite mission.
    """
    fig = plt.figure(figsize=(9, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    smax = max(1.6, np.nanmax(table['std_model'] / table['std_obs']) * 1.1)
    ax.set_ylim(0, smax)
    corr_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(corr_ticks))
    ax.set_xticklabels([f'{c:g}' for c in corr_ticks])
    ax.text(np.pi / 4, smax * 1.12, 'Correlation', rotation=-45, ha='center', va='center', fontsize=11)
    ax.set_ylabel('Normalised standard deviation', fontsize=11, labelpad=25)
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(theta, np.ones_like(theta), 'k--', lw=1)
    for r in (0.25, 0.5, 0.75, 1.0, 1.25):  # centred RMS difference isolines
        x = 1 + r * np.cos(np.linspace(0, np.pi, 200))
        y = r * np.sin(np.linspace(0, np.pi, 200))
        rr, tt = np.hypot(x, y), np.arctan2(y, x)
        keep = (tt <= np.pi / 2) & (rr <= smax)
        ax.plot(tt[keep], rr[keep], color='gray', lw=0.6, alpha=0.6)
    ax.plot(0, 1, 'k*', ms=15, label='Observations', clip_on=False)

    satellites = [s for s in table.index.get_level_values('satellite').unique() if s != 'ALL']
    for i, model in enumerate(table.index.get_level_values('model').unique()):
        color = COLORS[i % len(COLORS)]
        for j, sat in enumerate(satellites):
            row = table.loc[(model, sat)]
            if row['n'] > 10:
                ax.plot(np.arccos(np.clip(row['correlation'], 0, 1)), row['std_model'] / row['std_obs'],
                        MARKERS[j % len(MARKERS)], color=color, ms=7, alpha=0.6, mec='k', mew=0.4)
        row = table.loc[(model, 'ALL')]
        ax.plot(np.arccos(np.clip(row['correlation'], 0, 1)), row['std_model'] / row['std_obs'],
                'o', color=color, ms=14, mec='k', label=model)
    for j, sat in enumerate(satellites):
        ax.plot([], [], MARKERS[j % len(MARKERS)], color='gray', ms=7, label=sat)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.05), fontsize=9)
    ax.set_title(f'Taylor diagram {title}', fontsize=14, fontweight='bold', pad=20)
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_binned_errors(pairs, save_path: str, bin_width: float = 0.5, title: str = '') -> pd.DataFrame:
    """Bias and RMSE as a function of observed Hs (conditional on observations)."""
    from stats import grouped_metrics
    obs = pairs.hs.values
    edges = np.arange(0, np.ceil(obs.max() / bin_width) * bin_width + bin_width, bin_width)
    groups = np.clip(np.searchsorted(edges, obs, side='right') - 1, 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, (ax_b, ax_r, ax_n) = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                                           gridspec_kw={'height_ratios': [2, 2, 1]})
    rows = []
    for i, model in enumerate(pairs.model.values):
        res = grouped_metrics(pairs.model_hs.sel(model=model).values, obs, groups, len(centers), min_obs=5)
        color = COLORS[i % len(COLORS)]
        ax_b.plot(centers, res['bias'], 'o-', color=color, label=model)
        ax_r.plot(centers, res['rmse'], 'o-', color=color, label=model)
        rows.append(pd.DataFrame(res, index=pd.Index(centers, name='hs_bin_center')).assign(model=model))
    ax_n.bar(centers, res['nobs'], width=bin_width * 0.9, color='gray')
    ax_b.axhline(0, color='k', lw=1)
    for ax, label in ((ax_b, 'Bias [m]'), (ax_r, 'RMSE [m]'), (ax_n, 'N pairs')):
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
    ax_b.legend()
    ax_n.set_xlabel('Observed Hs [m]', fontsize=12)
    ax_b.set_title(f'Errors by observed Hs class {title}', fontsize=14, fontweight='bold')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return pd.concat(rows)


def main(conf_path: str, start_date: str, end_date: str) -> None:
    """
    Scatter plots, statistics tables, Taylor diagram and Hs-class errors for all
    experiments, computed on the common filtered sample (see pairs.load_pairs).
    """
    conf = load_plot_conf(conf_path)
    outdir = plot_dir(conf)
    date = f"{start_date}_{end_date}"
    title = conf.title.format(start_date=start_date, end_date=end_date)
    percentile_thresholds = conf.filters.get('percentile_thresholds') or []
    dpi = int(conf.get('dpi', 300))

    pairs = load_pairs(conf, start_date, end_date)
    sat2plot = pairs.hs.values

    for dataset in pairs.model.values:
        print(f"\nExperiment: {dataset}")
        mod2plot = pairs.model_hs.sel(model=dataset).values
        outName = os.path.join(outdir, f'scatter_{dataset}_{date}.jpeg')
        fig = scatter_waves(mod2plot, sat2plot, model_name=dataset, sat_name="Satellite",
                            save_path=outName, dpi=dpi, percentile_thresholds=percentile_thresholds,
                            include_qq_plot=True)
        plt.close(fig)
        fig = simple_scatter_plot(mod2plot, sat2plot, model_name=dataset, sat_name="Satellite",
                                  save_path=outName.replace(f'scatter_{dataset}', f'scatter_{dataset}_simple'),
                                  dpi=dpi)
        plt.close(fig)
        for perc in percentile_thresholds:
            try:
                fig = generate_percentile_scatter(
                    mod2plot, sat2plot, percentile=perc, model_name=dataset, sat_name="Satellite",
                    save_path=outName.replace(f'scatter_{dataset}', f'scatter_{dataset}_P{perc}'), dpi=dpi)
                plt.close(fig)
            except ValueError as e:
                print(f"  P{perc} scatter skipped: {e}")

    table = metrics_table(pairs)
    table.to_csv(os.path.join(outdir, f'metrics_{date}.csv'), float_format='%.4f')
    print('\n' + table[['n', 'bias', 'rmse', 'nbias', 'si', 'correlation']].round(3).to_string())
    taylor_diagram(table, os.path.join(outdir, f'taylor_{date}.jpeg'), title)
    plot_binned_errors(pairs, os.path.join(outdir, f'errors_by_hs_{date}.jpeg'), title=title).to_csv(
        os.path.join(outdir, f'errors_by_hs_{date}.csv'), float_format='%.4f')
    if 'unbiased' in pairs.attrs:
        print(pairs.attrs['unbiased'])
    print(f"\nOutput directory: {outdir}")
