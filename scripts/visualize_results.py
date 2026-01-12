#!/usr/bin/env python3
"""Visualization scripts for LTH Multi-Task RL results.

Generates publication-quality figures for:
1. Learning curves (return/success vs steps)
2. Mask structural analysis (Jaccard heatmaps)
3. Layerwise sparsity and overlap patterns
4. Shared core analysis

Usage:
    python scripts/visualize_results.py --results-dir results/lth_run_XXXXXX
    python scripts/visualize_results.py --results-dir results/lth_run_XXXXXX --output-dir figures
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (colorblind-friendly)
COLORS = {
    'push-v3': '#2274A5',      # Blue
    'reach-v3': '#E83151',     # Red
    'pick-place-v3': '#32936F', # Green
    'union': '#9B59B6',        # Purple
    'multitask': '#E67E22',    # Orange
    'shared': '#34495E',       # Dark gray
}


def load_metrics_history(results_dir: str, task_name: str, seed: int) -> List[Dict]:
    """Load metrics history from JSON file."""
    path = os.path.join(
        results_dir, "single_task", task_name, f"seed_{seed}", "metrics_history.json"
    )
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def load_structural_metrics(results_dir: str) -> Optional[Dict]:
    """Load structural analysis metrics."""
    path = os.path.join(results_dir, "analysis", "structural_metrics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_shared_core(results_dir: str) -> Optional[Dict]:
    """Load shared core analysis."""
    path = os.path.join(results_dir, "analysis", "shared_core.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_mask_summary(results_dir: str) -> Optional[Dict]:
    """Load mask summary."""
    path = os.path.join(results_dir, "analysis", "mask_summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def plot_learning_curves(
    results_dir: str,
    tasks: List[str],
    seed: int,
    output_path: str,
    metric: str = "mean_return",
) -> None:
    """Plot learning curves for all tasks.
    
    Args:
        results_dir: Directory containing results.
        tasks: List of task names.
        seed: Random seed.
        output_path: Path to save the figure.
        metric: Metric to plot ('mean_return' or 'mean_success').
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if not history:
            continue
        
        steps = [h["step"] for h in history]
        values = [h[metric] for h in history]
        
        color = COLORS.get(task, '#333333')
        ax.plot(steps, values, label=task.replace("-v3", "").title(), 
                color=color, linewidth=2)
        
        # Add shaded region for std if available
        if f"std_{metric.replace('mean_', '')}" in history[0]:
            stds = [h[f"std_{metric.replace('mean_', '')}"] for h in history]
            values = np.array(values)
            stds = np.array(stds)
            ax.fill_between(steps, values - stds, values + stds, 
                           alpha=0.2, color=color)
    
    # Formatting
    ax.set_xlabel("Training Steps")
    ylabel = "Episode Return" if "return" in metric else "Success Rate"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Learning Curves - {ylabel}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_success_curves(
    results_dir: str,
    tasks: List[str],
    seed: int,
    output_path: str,
) -> None:
    """Plot success rate curves for all tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if not history:
            continue
        
        steps = [h["step"] for h in history]
        success = [h["mean_success"] for h in history]
        
        color = COLORS.get(task, '#333333')
        ax.plot(steps, success, label=task.replace("-v3", "").title(),
                color=color, linewidth=2)
    
    # Add target line
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target (80%)')
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate Over Training")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_jaccard_heatmap(
    structural_metrics: Dict,
    tasks: List[str],
    output_path: str,
) -> None:
    """Plot Jaccard similarity heatmap between task masks."""
    n_tasks = len(tasks)
    similarity_matrix = np.eye(n_tasks)
    
    pairwise = structural_metrics.get("pairwise_jaccard", {})
    
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i < j:
                key = f"{task1}_vs_{task2}"
                if key in pairwise:
                    similarity_matrix[i, j] = pairwise[key]
                    similarity_matrix[j, i] = pairwise[key]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jaccard Similarity')
    
    # Labels
    task_labels = [t.replace("-v3", "").title() for t in tasks]
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(task_labels, rotation=45, ha='right')
    ax.set_yticklabels(task_labels)
    
    # Add values in cells
    for i in range(n_tasks):
        for j in range(n_tasks):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=12,
                          color='white' if similarity_matrix[i, j] < 0.5 else 'black')
    
    ax.set_title("Mask Similarity (Jaccard Index)")
    
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_layerwise_similarity(
    structural_metrics: Dict,
    tasks: List[str],
    output_path: str,
) -> None:
    """Plot layerwise Jaccard similarity as grouped bar chart."""
    layerwise = structural_metrics.get("layerwise_patterns", {})
    
    if not layerwise:
        print("  No layerwise data found, skipping...")
        return
    
    # Get first comparison to extract layer names
    first_key = list(layerwise.keys())[0]
    layers = list(layerwise[first_key].keys())
    
    # Simplify layer names for display
    layer_labels = [l.split("/")[-1] for l in layers]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.8 / len(layerwise)
    
    for idx, (comparison, values) in enumerate(layerwise.items()):
        similarities = [values.get(layer, 0) for layer in layers]
        offset = (idx - len(layerwise)/2 + 0.5) * width
        
        # Parse comparison name for label
        parts = comparison.split("_vs_")
        label = f"{parts[0]} vs {parts[1]}".replace("-v3", "")
        
        ax.bar(x + offset, similarities, width, label=label.title(), alpha=0.8)
    
    ax.set_xlabel("Network Layer")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Layerwise Mask Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sparsity_distribution(
    mask_summary: Dict,
    output_path: str,
) -> None:
    """Plot sparsity distribution across tasks."""
    tasks = list(mask_summary.keys())
    actor_sparsities = [mask_summary[t]["actor_sparsity"] for t in tasks]
    critic_sparsities = [mask_summary[t]["critic_sparsity"] for t in tasks]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    task_labels = [t.replace("-v3", "").title() for t in tasks]
    colors = [COLORS.get(t, '#333333') for t in tasks]
    
    bars1 = ax.bar(x - width/2, actor_sparsities, width, label='Actor',
                   color=[c for c in colors], alpha=0.8)
    bars2 = ax.bar(x + width/2, critic_sparsities, width, label='Critic',
                   color=[c for c in colors], alpha=0.5, hatch='//')
    
    # Add target line
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
    
    ax.set_xlabel("Task")
    ax.set_ylabel("Sparsity")
    ax.set_title("Network Sparsity by Task")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shared_core_analysis(
    shared_core: Dict,
    tasks: List[str],
    output_path: str,
) -> None:
    """Plot shared core analysis as pie chart and bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Pie chart of shared vs unique weights
    ax1 = axes[0]
    shared_frac = shared_core.get("shared_core_fraction", 0)
    unique_frac = 1 - shared_frac
    
    sizes = [shared_frac, unique_frac]
    labels = ['Shared Core', 'Task-Specific']
    colors_pie = [COLORS['shared'], '#95A5A6']
    explode = (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', shadow=False, startangle=90)
    ax1.set_title("Union Mask Composition")
    
    # Right: Bar chart of per-task shared fraction
    ax2 = axes[1]
    task_labels = [t.replace("-v3", "").title() for t in tasks]
    shared_fracs = [shared_core.get(f"{t}_in_shared", 0) for t in tasks]
    colors_bar = [COLORS.get(t, '#333333') for t in tasks]
    
    bars = ax2.bar(task_labels, shared_fracs, color=colors_bar, alpha=0.8)
    
    ax2.set_xlabel("Task")
    ax2.set_ylabel("Fraction in Shared Core")
    ax2.set_title("Per-Task Contribution to Shared Core")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.suptitle("Shared Core Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_training_efficiency(
    results_dir: str,
    tasks: List[str],
    seed: int,
    output_path: str,
) -> None:
    """Plot training efficiency metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Time to success threshold
    ax1 = axes[0]
    success_threshold = 0.8
    
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if not history:
            continue
        
        steps = [h["step"] for h in history]
        success = [h["mean_success"] for h in history]
        
        # Find first step above threshold
        step_to_success = None
        for s, succ in zip(steps, success):
            if succ >= success_threshold:
                step_to_success = s
                break
        
        color = COLORS.get(task, '#333333')
        label = task.replace("-v3", "").title()
        
        ax1.plot(steps, success, label=label, color=color, linewidth=2)
        
        if step_to_success:
            ax1.axvline(x=step_to_success, color=color, linestyle=':', alpha=0.5)
            ax1.scatter([step_to_success], [success_threshold], color=color, 
                       s=100, zorder=5, marker='*')
    
    ax1.axhline(y=success_threshold, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Success Rate")
    ax1.set_title(f"Time to {success_threshold:.0%} Success")
    ax1.legend(loc="lower right")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Right: Alpha (temperature) over time
    ax2 = axes[1]
    
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if not history:
            continue
        
        steps = [h["step"] for h in history]
        alpha = [h.get("alpha", 1.0) for h in history]
        
        color = COLORS.get(task, '#333333')
        label = task.replace("-v3", "").title()
        
        ax2.plot(steps, alpha, label=label, color=color, linewidth=2)
    
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Alpha (Temperature)")
    ax2.set_title("Entropy Coefficient Over Training")
    ax2.legend()
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_summary_figure(
    results_dir: str,
    tasks: List[str],
    seed: int,
    output_path: str,
) -> None:
    """Generate a summary figure combining key visualizations."""
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top-left: Learning curves
    ax1 = fig.add_subplot(gs[0, 0])
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if history:
            steps = [h["step"] for h in history]
            returns = [h["mean_return"] for h in history]
            color = COLORS.get(task, '#333333')
            ax1.plot(steps, returns, label=task.replace("-v3", "").title(),
                    color=color, linewidth=2)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Return")
    ax1.set_title("(a) Learning Curves")
    ax1.legend(loc="lower right")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Top-right: Success rate
    ax2 = fig.add_subplot(gs[0, 1])
    for task in tasks:
        history = load_metrics_history(results_dir, task, seed)
        if history:
            steps = [h["step"] for h in history]
            success = [h["mean_success"] for h in history]
            color = COLORS.get(task, '#333333')
            ax2.plot(steps, success, label=task.replace("-v3", "").title(),
                    color=color, linewidth=2)
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Success Rate")
    ax2.set_title("(b) Success Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Bottom-left: Sparsity
    ax3 = fig.add_subplot(gs[1, 0])
    mask_summary = load_mask_summary(results_dir)
    if mask_summary:
        task_labels = [t.replace("-v3", "").title() for t in tasks if t in mask_summary]
        actor_sp = [mask_summary[t]["actor_sparsity"] for t in tasks if t in mask_summary]
        colors_bar = [COLORS.get(t, '#333333') for t in tasks if t in mask_summary]
        
        if task_labels:
            ax3.bar(task_labels, actor_sp, color=colors_bar, alpha=0.8)
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
            ax3.set_ylabel("Sparsity")
            ax3.set_title("(c) Achieved Sparsity (Actor)")
            ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, "No sparsity data", ha='center', va='center', 
                transform=ax3.transAxes)
        ax3.set_title("(c) Achieved Sparsity")
    
    # Bottom-right: Shared core (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    shared_core = load_shared_core(results_dir)
    if shared_core and len(tasks) > 1:
        shared_frac = shared_core.get("shared_core_fraction", 0)
        ax4.pie([shared_frac, 1-shared_frac], 
               labels=['Shared', 'Unique'],
               colors=[COLORS['shared'], '#95A5A6'],
               autopct='%1.1f%%', startangle=90)
        ax4.set_title("(d) Shared Core Fraction")
    else:
        ax4.text(0.5, 0.5, "Single task\n(no shared core)", ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title("(d) Shared Core Analysis")
    
    plt.suptitle("LTH Multi-Task RL: Training Summary", fontsize=16, y=0.98)
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LTH results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing LTH results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save figures (default: results-dir/figures)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to visualize")
    parser.add_argument("--tasks", type=str, nargs="+", 
                        default=["push-v3"],
                        help="Tasks to visualize")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    print(f"  Results: {args.results_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Tasks:   {args.tasks}")
    print()
    
    # Generate figures
    print("Generating learning curves...")
    plot_learning_curves(
        args.results_dir, args.tasks, args.seed,
        os.path.join(args.output_dir, "learning_curves.png")
    )
    
    print("Generating success rate plot...")
    plot_success_curves(
        args.results_dir, args.tasks, args.seed,
        os.path.join(args.output_dir, "success_curves.png")
    )
    
    print("Generating training efficiency plot...")
    plot_training_efficiency(
        args.results_dir, args.tasks, args.seed,
        os.path.join(args.output_dir, "training_efficiency.png")
    )
    
    # Structural analysis (if available)
    structural_metrics = load_structural_metrics(args.results_dir)
    if structural_metrics:
        print("Generating Jaccard heatmap...")
        plot_jaccard_heatmap(
            structural_metrics, args.tasks,
            os.path.join(args.output_dir, "jaccard_heatmap.png")
        )
        
        print("Generating layerwise similarity...")
        plot_layerwise_similarity(
            structural_metrics, args.tasks,
            os.path.join(args.output_dir, "layerwise_similarity.png")
        )
    
    # Mask summary (if available)
    mask_summary = load_mask_summary(args.results_dir)
    if mask_summary:
        print("Generating sparsity distribution...")
        plot_sparsity_distribution(
            mask_summary,
            os.path.join(args.output_dir, "sparsity_distribution.png")
        )
    
    # Shared core (if available)
    shared_core = load_shared_core(args.results_dir)
    if shared_core:
        print("Generating shared core analysis...")
        plot_shared_core_analysis(
            shared_core, args.tasks,
            os.path.join(args.output_dir, "shared_core_analysis.png")
        )
    
    # Summary figure
    print("Generating summary figure...")
    generate_summary_figure(
        args.results_dir, args.tasks, args.seed,
        os.path.join(args.output_dir, "summary.png")
    )
    
    print(f"\n{'='*60}")
    print(f"All figures saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
