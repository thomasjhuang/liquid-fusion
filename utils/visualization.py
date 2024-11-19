import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

def plot_attention_comparison(results: Dict[str, Dict[str, float]], 
                            save_dir: str = "figures",
                            save_format: str = "png",
                            style: str = "whitegrid") -> None:
    """
    Create comprehensive visualizations comparing different attention mechanisms.
    
    Args:
        results: Dictionary of results with structure {attention_type: {metric: value}}
        save_dir: Directory to save the generated plots
        save_format: Format to save the plots (png, pdf, svg)
        style: Seaborn style to use for plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    sns.set_theme(style=style)
    plt.rcParams['figure.figsize'] = (12, 8)
    
    data = []
    for attn_type, metrics in results.items():
        for metric, value in metrics.items():
            data.append({
                'Attention Type': attn_type,
                'Metric': metric,
                'Value': value
            })
    df = pd.DataFrame(data)
    
    # 1. Main Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Attention Mechanisms', fontsize=16, y=1.02)
    
    metrics = ['perplexity', 'memory_usage', 'inference_time', 'attention_sparsity']
    titles = ['Perplexity (lower is better)', 
             'Memory Usage (MB)', 
             'Inference Time (s)',
             'Attention Sparsity (higher is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        metric_data = df[df['Metric'] == metric]

        sns.barplot(
            x='Attention Type',
            y='Value',
            data=metric_data,
            ax=ax,
            palette='husl'
        )
    
        for i, v in enumerate(metric_data['Value']):
            ax.text(
                i, v, 
                f'{v:.3f}', 
                ha='center', 
                va='bottom'
            )
        ax.set_title(title, pad=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if metric == 'perplexity':
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path / f'attention_comparison.{save_format}', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Memory-Performance Trade-off Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df[df['Metric'].isin(['memory_usage', 'perplexity'])],
        x='Value',
        y='Value',
        hue='Attention Type',
        style='Attention Type',
        s=100
    )
    
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Perplexity (log scale)')
    plt.yscale('log')
    plt.title('Memory-Performance Trade-off')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path / f'memory_performance_tradeoff.{save_format}', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Relative Improvement Plot
    baseline_metrics = {metric: results['baseline'][metric] for metric in metrics}
    improvements = {}
    
    for attn_type in ['liquidfusion', 'h2o', 'streaming']:
        improvements[attn_type] = {}
        for metric in metrics:
            if baseline_metrics[metric] != 0:
                rel_improvement = ((results[attn_type][metric] - baseline_metrics[metric]) 
                                 / baseline_metrics[metric] * 100)
                improvements[attn_type][metric] = rel_improvement
    
    imp_data = []
    for attn_type, metrics_imp in improvements.items():
        for metric, imp in metrics_imp.items():
            imp_data.append({
                'Attention Type': attn_type,
                'Metric': metric,
                'Improvement (%)': imp
            })
    imp_df = pd.DataFrame(imp_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Metric',
        y='Improvement (%)',
        hue='Attention Type',
        data=imp_df,
        palette='husl'
    )
    
    plt.title('Relative Improvement over Baseline (%)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Attention Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path / f'relative_improvements.{save_format}', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    summary_df = pd.DataFrame(results).round(3)
    summary_df.to_csv(save_path / 'attention_comparison_results.csv')
    
    styled_table = summary_df.style\
        .highlight_min(color='lightgreen', axis=1)\
        .highlight_max(color='lightcoral', axis=1)\
        .format(precision=3)
    
    styled_table.to_html(save_path / 'attention_comparison_results.html')

    print("\nSummary Statistics:")
    print("-" * 50)
    print(summary_df.to_string())
    print("\nRelative Improvements:")
    print("-" * 50)
    for attn_type, metrics_imp in improvements.items():
        print(f"\n{attn_type.upper()}:")
        for metric, imp in metrics_imp.items():
            better = "better" if (metric in ['attention_sparsity'] and imp > 0) or \
                                (metric not in ['attention_sparsity'] and imp < 0) else "worse"
            print(f"{metric}: {imp:.2f}% ({better})")

def plot_attention_patterns(attention_weights: torch.Tensor,
                          save_path: str = "attention_patterns.png"):
    """
    Plot attention patterns for different attention mechanisms.
    
    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
        save_path: Path to save the visualization
    """
    avg_attention = attention_weights.mean(dim=(0, 1)).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention,
        cmap='viridis',
        xticklabels=False,
        yticklabels=False
    )
    plt.title('Average Attention Pattern')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Attention Weight')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
