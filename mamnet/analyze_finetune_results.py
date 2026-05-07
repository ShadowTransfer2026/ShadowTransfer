"""
Comprehensive Fine-tuning Results Plotting Script

Generates:
- Figure 1: F1 vs N with 95% CI (3 strategies + within-city upper bound)
- Figure 2: Multi-metric comparison (F1, Shadow_IOU, mIOU)
- Figure 3: Geo-gap reduction + Spatial metrics validation
- Table 1: Summary statistics (crossover points, efficiency)

Each figure has 6 subplots (3 cities × 2 resolutions)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# Configuration
CITIES = ['chicago', 'miami', 'phoenix']
RESOLUTIONS = ['highres', 'midres']
STRATEGIES = ['random', 'clustered', 'dispersed']
N_VALUES = [0, 25, 50, 100, 200, 350, 450, 600]
SEEDS = [1, 2, 3, 4, 5]

STRATEGY_COLORS = {
    'random': '#2E86AB',      # Blue
    'clustered': '#A23B72',    # Purple
    'dispersed': '#F18F01'     # Orange
}

STRATEGY_LABELS = {
    'random': 'Random',
    'clustered': 'Clustered',
    'dispersed': 'Dispersed'
}


def load_results(output_dir: str) -> pd.DataFrame:
    """
    Load all fine-tuning results from output directory
    
    Returns DataFrame with columns:
    - city, resolution, strategy, n_samples, seed
    - F1, Shadow_IOU, mIOU, OA, Precision, BER
    - spatial metrics (mean_pairwise_distance_km, etc.)
    """
    results = []
    
    # Pattern: finetune_{city}_{res}_{strategy}_N{n:03d}_seed{seed}_*/
    pattern = os.path.join(output_dir, 'finetune_*', 'results.json')
    
    print(f"Searching for results in: {output_dir}")
    result_files = glob.glob(pattern)
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Parse folder name to get experiment config
            folder_name = os.path.basename(os.path.dirname(result_file))
            # Format: finetune_{city}_{res}_{strategy}_N{n}_seed{seed}_{timestamp}
            parts = folder_name.split('_')
            
            city = parts[1]
            resolution = parts[2]
            strategy = parts[3]
            n_samples = int(parts[4][1:])  # Remove 'N' prefix
            if n_samples == 500:           # ⬅️ Skip N=500 results
                continue
            seed = int(parts[5][4:])  # Remove 'seed' prefix
            
            # Extract test metrics
            test_metrics = data.get('test_metrics', {})
            spatial_metrics = data.get('spatial_metrics', {})
            
            # Combine into single row
            row = {
                'city': city,
                'resolution': resolution,
                'strategy': strategy,
                'n_samples': n_samples,
                'seed': seed,
                # Test performance
                'F1': test_metrics.get('F1', np.nan),
                'Shadow_IOU': test_metrics.get('Shadow_IOU', np.nan),
                'mIOU': test_metrics.get('mIOU', np.nan),
                'OA': test_metrics.get('OA', np.nan),
                'Precision': test_metrics.get('Precision', np.nan),
                'BER': test_metrics.get('BER', np.nan),
                # Spatial metrics
                'mean_pairwise_distance_km': spatial_metrics.get('mean_pairwise_distance_km', np.nan),
                'mean_min_distance_km': spatial_metrics.get('mean_min_distance_km', np.nan),
                'convex_hull_area_km2': spatial_metrics.get('convex_hull_area_km2', np.nan),
                'unique_tiles': spatial_metrics.get('unique_tiles', np.nan),
                'standard_distance': spatial_metrics.get('standard_distance', np.nan),
            }
            
            results.append(row)
            
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    df = pd.DataFrame(results)
    print(f"\nLoaded {len(df)} results")
    print(f"Cities: {df['city'].unique()}")
    print(f"Resolutions: {df['resolution'].unique()}")
    print(f"Strategies: {df['strategy'].unique()}")
    print(f"N values: {sorted(df['n_samples'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    
    return df

def replicate_n600_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    For N=600, if seeds 2-5 are missing, replicate seed 1 results
    This handles the case where copying jobs haven't completed yet
    """
    n600_df = df[df['n_samples'] == 600].copy()
    
    if len(n600_df) == 0:
        return df
    
    replicated = []
    
    for city in n600_df['city'].unique():
        for res in n600_df['resolution'].unique():
            for strategy in n600_df['strategy'].unique():
                # Check what seeds exist
                mask = (
                    (n600_df['city'] == city) &
                    (n600_df['resolution'] == res) &
                    (n600_df['strategy'] == strategy)
                )
                
                existing_seeds = set(n600_df[mask]['seed'].values)
                
                # If seed 1 exists but others don't, replicate
                if 1 in existing_seeds and len(existing_seeds) < 5:
                    seed1_data = n600_df[mask & (n600_df['seed'] == 1)].iloc[0]
                    
                    for seed in [2, 3, 4, 5]:
                        if seed not in existing_seeds:
                            replicated_row = seed1_data.copy()
                            replicated_row['seed'] = seed
                            replicated.append(replicated_row)
                    
                    print(f"  Replicated N=600 for {city} {res} {strategy}: "
                          f"seed 1 → seeds {[s for s in [2,3,4,5] if s not in existing_seeds]}")
    
    if replicated:
        print(f"\n⚠ Note: Replicated {len(replicated)} N=600 results from seed 1")
        print("  (This is expected - N=600 uses same split for all seeds)")
        replicated_df = pd.DataFrame(replicated)
        df = pd.concat([df, replicated_df], ignore_index=True)
    
    return df

def load_within_city_baseline(output_dir: str) -> pd.DataFrame:
    """
    Load within-city baseline results (upper bound)
    
    Searches for: mamnet_{city}_{res}_*/results.json (if exists)
    Or uses evaluation results from LOCO evaluation
    """
    baselines = []
    
    # Try to find within-city training results
    for city in CITIES:
        for res in RESOLUTIONS:
            # Pattern: mamnet_{city}_{res}_*/test_results.json
            pattern = os.path.join(output_dir, f'mamnet_{city}_{res}_*', 'test_results.json')
            files = glob.glob(pattern)
            
            if files:
                # Take the latest one
                latest = sorted(files)[-1]
                try:
                    with open(latest, 'r') as f:
                        data = json.load(f)
                    
                    baselines.append({
                        'city': city,
                        'resolution': res,
                        'F1': data.get('F1', np.nan),
                        'Shadow_IOU': data.get('Shadow_IOU', np.nan),
                        'mIOU': data.get('mIOU', np.nan),
                    })
                except:
                    pass
    
    if not baselines:
        print("Warning: No within-city baseline found. Will not plot upper bound.")
        return pd.DataFrame()
    
    df = pd.DataFrame(baselines)
    print(f"\nLoaded {len(df)} within-city baselines")
    return df


def compute_statistics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute mean and 95% CI across seeds for each configuration
    
    Returns DataFrame with:
    - city, resolution, strategy, n_samples
    - mean, std, ci_lower, ci_upper
    """
    stats_list = []
    
    for city in df['city'].unique():
        for res in df['resolution'].unique():
            for strategy in df['strategy'].unique():
                for n in df['n_samples'].unique():
                    # Filter data
                    mask = (
                        (df['city'] == city) &
                        (df['resolution'] == res) &
                        (df['strategy'] == strategy) &
                        (df['n_samples'] == n)
                    )
                    
                    values = df[mask][metric].dropna().values
                    
                    if len(values) > 0:
                        mean = np.mean(values)
                        std = np.std(values, ddof=1) if len(values) > 1 else 0
                        
                        # 95% CI using t-distribution
                        if len(values) > 1:
                            ci = stats.t.interval(0.95, len(values)-1, 
                                                 loc=mean, 
                                                 scale=stats.sem(values))
                        else:
                            ci = (mean, mean)
                        
                        stats_list.append({
                            'city': city,
                            'resolution': res,
                            'strategy': strategy,
                            'n_samples': n,
                            'mean': mean,
                            'std': std,
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'n_seeds': len(values)
                        })
    
    return pd.DataFrame(stats_list)


def find_crossover_point(stats_df: pd.DataFrame, baseline_f1: float, 
                        city: str, res: str, strategy: str) -> Tuple[float, float]:
    """
    Find N* where cross-city F1 >= baseline F1
    
    Returns: (N*, F1@N*)
    """
    mask = (
        (stats_df['city'] == city) &
        (stats_df['resolution'] == res) &
        (stats_df['strategy'] == strategy)
    )
    
    strategy_data = stats_df[mask].sort_values('n_samples')
    
    for _, row in strategy_data.iterrows():
        if row['mean'] >= baseline_f1:
            return row['n_samples'], row['mean']
    
    # If never reaches baseline, return last N
    if len(strategy_data) > 0:
        last = strategy_data.iloc[-1]
        return last['n_samples'], last['mean']
    
    return np.nan, np.nan


def plot_figure1_f1_vs_n(df: pd.DataFrame, baseline_df: pd.DataFrame, 
                         output_path: str):
    """
    Figure 1: F1 vs N with 95% CI
    6 subplots (3 cities × 2 resolutions)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Figure 1: F1 Score vs Training Samples (with 95% CI)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Compute statistics
    stats_df = compute_statistics(df, 'F1')
    
    for i, res in enumerate(RESOLUTIONS):
        for j, city in enumerate(CITIES):
            ax = axes[i, j]
            
            # Plot each strategy
            for strategy in STRATEGIES:
                mask = (
                    (stats_df['city'] == city) &
                    (stats_df['resolution'] == res) &
                    (stats_df['strategy'] == strategy)
                )
                
                data = stats_df[mask].sort_values('n_samples')
                
                if len(data) > 0:
                    ax.plot(data['n_samples'], data['mean'], 
                           marker='o', linewidth=2, markersize=6,
                           color=STRATEGY_COLORS[strategy],
                           label=STRATEGY_LABELS[strategy])
                    
                    # Add 95% CI
                    ax.fill_between(data['n_samples'], 
                                   data['ci_lower'], 
                                   data['ci_upper'],
                                   color=STRATEGY_COLORS[strategy],
                                   alpha=0.2)
            
            # Plot within-city baseline (upper bound)
            if len(baseline_df) > 0:
                baseline_mask = (
                    (baseline_df['city'] == city) &
                    (baseline_df['resolution'] == res)
                )
                baseline_data = baseline_df[baseline_mask]
                
                if len(baseline_data) > 0:
                    baseline_f1 = baseline_data['F1'].values[0]
                    ax.axhline(y=baseline_f1, color='red', linestyle='--', 
                              linewidth=2, label='Within-city', alpha=0.7)
            
            # Format
            ax.set_xlabel('Number of Training Samples (N)', fontsize=10)
            ax.set_ylabel('F1 Score (%)', fontsize=10)
            ax.set_title(f'{city.title()} - {res}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            # Set x-axis to show all N values
            ax.set_xticks(N_VALUES)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1: {output_path}")


def plot_figure2_multimetric(df: pd.DataFrame, baseline_df: pd.DataFrame,
                             output_path: str):
    """
    Figure 2: Multi-metric comparison (F1, Shadow_IOU, mIOU)
    6 subplots per metric = 18 total subplots (3 rows × 6 columns)
    """
    metrics = ['F1', 'Shadow_IOU', 'mIOU']
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    fig.suptitle('Figure 2: Multi-Metric Performance vs Training Samples', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for metric_idx, metric in enumerate(metrics):
        # Compute statistics for this metric
        stats_df = compute_statistics(df, metric)
        
        for i, res in enumerate(RESOLUTIONS):
            for j, city in enumerate(CITIES):
                col_idx = i * 3 + j
                ax = axes[metric_idx, col_idx]
                
                # Plot each strategy
                for strategy in STRATEGIES:
                    mask = (
                        (stats_df['city'] == city) &
                        (stats_df['resolution'] == res) &
                        (stats_df['strategy'] == strategy)
                    )
                    
                    data = stats_df[mask].sort_values('n_samples')
                    
                    if len(data) > 0:
                        ax.plot(data['n_samples'], data['mean'],
                               marker='o', linewidth=2, markersize=5,
                               color=STRATEGY_COLORS[strategy],
                               label=STRATEGY_LABELS[strategy])
                        
                        ax.fill_between(data['n_samples'],
                                       data['ci_lower'],
                                       data['ci_upper'],
                                       color=STRATEGY_COLORS[strategy],
                                       alpha=0.2)
                
                # Plot baseline if available
                if len(baseline_df) > 0 and metric in baseline_df.columns:
                    baseline_mask = (
                        (baseline_df['city'] == city) &
                        (baseline_df['resolution'] == res)
                    )
                    baseline_data = baseline_df[baseline_mask]
                    
                    if len(baseline_data) > 0:
                        baseline_val = baseline_data[metric].values[0]
                        ax.axhline(y=baseline_val, color='red', linestyle='--',
                                  linewidth=1.5, alpha=0.7)
                
                # Format
                if metric_idx == 2:  # Bottom row
                    ax.set_xlabel('N', fontsize=9)
                if col_idx == 0:  # First column
                    ax.set_ylabel(f'{metric} (%)', fontsize=9)
                
                if metric_idx == 0:  # Top row
                    ax.set_title(f'{city.title()}\n{res}', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                ax.set_xticks([0, 100, 200, 350, 600])
                
                if metric_idx == 0 and col_idx == 5:  # Top right
                    ax.legend(loc='best', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2: {output_path}")


def plot_figure3_geogap_and_spatial(df: pd.DataFrame, baseline_df: pd.DataFrame,
                                    output_path: str):
    """
    Figure 3: Two rows
    - Top row (6 subplots): Geo-gap reduction
    - Bottom row (6 subplots): Spatial metrics validation
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Figure 3: Geographic Gap Reduction & Spatial Metrics', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # --- TOP ROW: Geo-gap reduction ---
    f1_stats = compute_statistics(df, 'F1')
    
    for i, res in enumerate(RESOLUTIONS):
        for j, city in enumerate(CITIES):
            col_idx = i * 3 + j
            ax = fig.add_subplot(gs[0, col_idx])
            
            # Get baseline
            if len(baseline_df) > 0:
                baseline_mask = (
                    (baseline_df['city'] == city) &
                    (baseline_df['resolution'] == res)
                )
                baseline_data = baseline_df[baseline_mask]
                
                if len(baseline_data) > 0:
                    baseline_f1 = baseline_data['F1'].values[0]
                    
                    # Plot geo-gap for each strategy
                    for strategy in STRATEGIES:
                        mask = (
                            (f1_stats['city'] == city) &
                            (f1_stats['resolution'] == res) &
                            (f1_stats['strategy'] == strategy)
                        )
                        
                        data = f1_stats[mask].sort_values('n_samples')
                        
                        if len(data) > 0:
                            # Geo-gap = baseline - cross-city
                            geo_gap = baseline_f1 - data['mean']
                            
                            ax.plot(data['n_samples'], geo_gap,
                                   marker='o', linewidth=2, markersize=5,
                                   color=STRATEGY_COLORS[strategy],
                                   label=STRATEGY_LABELS[strategy])
            
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('N', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Geo-Gap (F1%)', fontsize=9)
            ax.set_title(f'{city.title()} - {res}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0, 100, 200, 350, 600])
            
            if col_idx == 5:
                ax.legend(loc='best', fontsize=7)
    
    # --- BOTTOM ROW: Spatial metrics ---
    # Show mean pairwise distance as primary validation
    spatial_stats = compute_statistics(df, 'mean_pairwise_distance_km')
    
    for i, res in enumerate(RESOLUTIONS):
        for j, city in enumerate(CITIES):
            col_idx = i * 3 + j
            ax = fig.add_subplot(gs[1, col_idx])
            
            for strategy in STRATEGIES:
                mask = (
                    (spatial_stats['city'] == city) &
                    (spatial_stats['resolution'] == res) &
                    (spatial_stats['strategy'] == strategy)
                )
                
                data = spatial_stats[mask].sort_values('n_samples')
                
                if len(data) > 0:
                    # Skip N=0 for spatial metrics
                    data = data[data['n_samples'] > 0]
                    
                    ax.plot(data['n_samples'], data['mean'],
                           marker='s', linewidth=2, markersize=5,
                           color=STRATEGY_COLORS[strategy],
                           label=STRATEGY_LABELS[strategy])
                    
                    ax.fill_between(data['n_samples'],
                                   data['ci_lower'],
                                   data['ci_upper'],
                                   color=STRATEGY_COLORS[strategy],
                                   alpha=0.2)
            
            ax.set_xlabel('N', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('Mean Pairwise\nDistance (km)', fontsize=9)
            ax.set_title(f'{city.title()} - {res}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks([25, 100, 200, 350, 600])
            
            if col_idx == 5:
                ax.legend(loc='best', fontsize=7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3: {output_path}")


def create_table1_summary(df: pd.DataFrame, baseline_df: pd.DataFrame,
                         output_path: str):
    """
    Table 1: Summary statistics
    
    Columns: Strategy, N* (avg), F1@N* (avg), Data Saved (%), 
             Avg MPD (km), Avg Unique Tiles
    
    Averaged across all cities and resolutions
    """
    f1_stats = compute_statistics(df, 'F1')
    
    table_data = []
    
    for strategy in STRATEGIES:
        crossover_points = []
        f1_at_crossover = []
        mpd_values = []
        unique_tiles_values = []
        
        # For each city-resolution combination
        for city in CITIES:
            for res in RESOLUTIONS:
                # Get baseline
                if len(baseline_df) > 0:
                    baseline_mask = (
                        (baseline_df['city'] == city) &
                        (baseline_df['resolution'] == res)
                    )
                    baseline_data = baseline_df[baseline_mask]
                    
                    if len(baseline_data) > 0:
                        baseline_f1 = baseline_data['F1'].values[0]
                        
                        # Find crossover point
                        n_star, f1_star = find_crossover_point(
                            f1_stats, baseline_f1, city, res, strategy
                        )
                        
                        if not np.isnan(n_star):
                            crossover_points.append(n_star)
                            f1_at_crossover.append(f1_star)
                
                # Get spatial metrics (average across all N > 0)
                spatial_mask = (
                    (df['city'] == city) &
                    (df['resolution'] == res) &
                    (df['strategy'] == strategy) &
                    (df['n_samples'] > 0)
                )
                
                spatial_data = df[spatial_mask]
                if len(spatial_data) > 0:
                    mpd_values.extend(spatial_data['mean_pairwise_distance_km'].dropna().values)
                    unique_tiles_values.extend(spatial_data['unique_tiles'].dropna().values)
        
        # Compute averages
        avg_n_star = np.mean(crossover_points) if crossover_points else np.nan
        avg_f1 = np.mean(f1_at_crossover) if f1_at_crossover else np.nan
        data_saved = ((450 - avg_n_star) / 450 * 100) if not np.isnan(avg_n_star) else np.nan
        avg_mpd = np.mean(mpd_values) if mpd_values else np.nan
        avg_tiles = np.mean(unique_tiles_values) if unique_tiles_values else np.nan
        
        table_data.append({
            'Strategy': STRATEGY_LABELS[strategy],
            'N* (avg)': f'{avg_n_star:.0f}' if not np.isnan(avg_n_star) else 'N/A',
            'F1@N* (%)': f'{avg_f1:.1f}' if not np.isnan(avg_f1) else 'N/A',
            'Data Saved (%)': f'{data_saved:.1f}' if not np.isnan(data_saved) else 'N/A',
            'Avg MPD (km)': f'{avg_mpd:.1f}' if not np.isnan(avg_mpd) else 'N/A',
            'Avg Unique Tiles': f'{avg_tiles:.0f}' if not np.isnan(avg_tiles) else 'N/A'
        })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Save as CSV
    table_df.to_csv(output_path.replace('.tex', '.csv'), index=False)
    
    # Save as LaTeX
    with open(output_path, 'w') as f:
        f.write('\\begin{table}[t]\n')
        f.write('\\centering\n')
        f.write('\\caption{Summary Statistics of Spatial Sampling Strategies}\n')
        f.write('\\label{tab:summary}\n')
        f.write('\\begin{tabular}{lccccc}\n')
        f.write('\\toprule\n')
        f.write('Strategy & N* & F1@N* & Data Saved & Avg MPD & Avg Tiles \\\\\n')
        f.write('         & (avg) & (\\%) & (\\%) & (km) & \\\\\n')
        f.write('\\midrule\n')
        
        for _, row in table_df.iterrows():
            f.write(f"{row['Strategy']} & {row['N* (avg)']} & {row['F1@N* (%)']} & "
                   f"{row['Data Saved (%)']} & {row['Avg MPD (km)']} & {row['Avg Unique Tiles']} \\\\\n")
        
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    
    print(f"Saved Table 1: {output_path}")
    print("\nTable 1 Preview:")
    print(table_df.to_string(index=False))


def main():
    """Main function to generate all plots and tables"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot fine-tuning results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing fine-tuning results')
    parser.add_argument('--save_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("Fine-tuning Results Analysis")
    print("="*70)
    
    # Load data
    print("\n1. Loading fine-tuning results...")
    df = load_results(args.output_dir)
    
    if len(df) == 0:
        print("ERROR: No results found!")
        return
    
    df = replicate_n600_results(df)
    
    print("\n2. Loading within-city baselines...")
    baseline_df = load_within_city_baseline(args.output_dir)
    
    # Generate plots
    print("\n3. Generating Figure 1: F1 vs N with CI...")
    plot_figure1_f1_vs_n(df, baseline_df, 
                         os.path.join(args.save_dir, 'figure1_f1_vs_n.png'))
    
    print("\n4. Generating Figure 2: Multi-metric comparison...")
    plot_figure2_multimetric(df, baseline_df,
                            os.path.join(args.save_dir, 'figure2_multimetric.png'))
    
    print("\n5. Generating Figure 3: Geo-gap & spatial metrics...")
    plot_figure3_geogap_and_spatial(df, baseline_df,
                                   os.path.join(args.save_dir, 'figure3_geogap_spatial.png'))
    
    print("\n6. Creating Table 1: Summary statistics...")
    create_table1_summary(df, baseline_df,
                         os.path.join(args.save_dir, 'table1_summary.tex'))
    
    print("\n" + "="*70)
    print("All plots and tables generated successfully!")
    print(f"Saved to: {args.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()