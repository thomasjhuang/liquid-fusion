# benchmark_utils.py
from typing import Dict, List
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def compare_perplexity(results1: Dict, results2: Dict, name1: str = "Config1", name2: str = "Config2") -> Dict:
    """
    Compare perplexity scores between two benchmark results.
    
    Args:
        results1: First benchmark results
        results2: Second benchmark results
        name1: Name for first config
        name2: Name for second config
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    for dataset in set(results1["datasets"].keys()) & set(results2["datasets"].keys()):
        if "perplexity" in results1["datasets"][dataset] and "perplexity" in results2["datasets"][dataset]:
            comparison[dataset] = {
                name1: results1["datasets"][dataset]["perplexity"],
                name2: results2["datasets"][dataset]["perplexity"]
            }
    return comparison

def compare_latency(results1: Dict, results2: Dict, name1: str = "Config1", name2: str = "Config2") -> Dict:
    """Compare latency metrics between two benchmark results."""
    comparison = {}
    metrics = ["avg_latency_ms", "p90_latency_ms", "throughput_tokens_sec"]
    
    for dataset in set(results1["datasets"].keys()) & set(results2["datasets"].keys()):
        comparison[dataset] = {}
        for metric in metrics:
            if metric in results1["datasets"][dataset] and metric in results2["datasets"][dataset]:
                comparison[dataset][metric] = {
                    name1: results1["datasets"][dataset][metric],
                    name2: results2["datasets"][dataset][metric]
                }
    return comparison

def plot_comparison(comparison: Dict, metric: str, title: str = None):
    """
    Plot comparison results.
    
    Args:
        comparison: Comparison dictionary
        metric: Metric to plot
        title: Plot title
    """
    df = pd.DataFrame(comparison).T
    ax = df.plot(kind='bar', figsize=(12, 6))
    plt.title(title or f"Comparison of {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax

def load_results(filename: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def save_results(results: Dict, prefix: str = "benchmark"):
    """Save benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    return filename
