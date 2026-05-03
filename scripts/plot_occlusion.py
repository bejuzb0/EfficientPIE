# scripts/plot_occlusion.py
"""
Reads results/occlusion_results.csv and produces two plots:
1. Bar chart: accuracy per fixed occlusion kind.
2. Line plot: accuracy vs. random-patch percentage.
Saves the combined figure to results/occlusion_plot.png.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    csv_path = 'results/occlusion_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please run the evaluation script first.")
        return

    df = pd.read_csv(csv_path)

    # 1. Separate fixed conditions and random conditions
    fixed_order = ['none', 'top_half', 'bottom_half', 'left_half', 'right_half', 'pedestrian_only', 'context_only']
    df_fixed = df[df['kind'].isin(fixed_order)].copy()
    # Sort strictly by the fixed order
    df_fixed['kind'] = pd.Categorical(df_fixed['kind'], categories=fixed_order, ordered=True)
    df_fixed = df_fixed.sort_values('kind')

    df_random = df[df['kind'].str.startswith('random_')].copy()
    
    # Extract percentage for random
    def parse_pct(k):
        try:
            return int(k.split('_')[1].replace('pct', ''))
        except:
            return 0
    
    df_random['pct'] = df_random['kind'].apply(parse_pct)
    df_random = df_random.sort_values('pct')

    # Baseline accuracy
    try:
        baseline_acc = df_fixed[df_fixed['kind'] == 'none']['accuracy'].iloc[0]
    except IndexError:
        print("Warning: 'none' baseline not found in the results.")
        baseline_acc = None

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Bar chart for fixed occlusion kinds ---
    bars = ax1.bar(df_fixed['kind'], df_fixed['accuracy'], color='skyblue', edgecolor='black')
    if baseline_acc is not None:
        ax1.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline ({baseline_acc:.3f})')
        ax1.legend()
    
    ax1.set_title('Accuracy vs. Fixed Occlusion', fontsize=14)
    ax1.set_xlabel('Occlusion Kind', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add exact values on top of bars
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # --- Plot 2: Line plot for random patches ---
    if not df_random.empty:
        # Check if std is present
        if 'accuracy_std' in df_random.columns:
            ax2.errorbar(df_random['pct'], df_random['accuracy'], yerr=df_random['accuracy_std'], 
                         fmt='-o', color='purple', capsize=5, capthick=1.5, ecolor='gray')
        else:
            ax2.plot(df_random['pct'], df_random['accuracy'], '-o', color='purple')
            
        if baseline_acc is not None:
            ax2.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
            ax2.legend()
            
        ax2.set_title('Accuracy vs. Random Patch Area', fontsize=14)
        ax2.set_xlabel('Occluded Area Percentage (%)', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax2.set_xticks(df_random['pct'])
    else:
        ax2.text(0.5, 0.5, 'No random patch data found', ha='center', va='center')

    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    out_path = 'results/occlusion_plot.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
