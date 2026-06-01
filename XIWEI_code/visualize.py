"""
Visualization tools for JSP Scheduling RL experiments.

Generates comparison charts from training logs and evaluation results.
Standalone: depends only on matplotlib, numpy, and JSON log files.

Usage:
    python visualize.py                          # scan logs/ dir and generate all charts
    python visualize.py --log logs/train_10x10x3_nstep3.json
    python visualize.py --compare logs/           # compare multiple training runs
    python visualize.py --save-dir charts/        # save charts to directory
"""

import os
import sys
import json
import argparse
import glob
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ═══════════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════════

def set_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# Color palette
COLORS = {
    'DQN': '#2196F3',
    'Greedy': '#4CAF50',
    'Random': '#FF9800',
    'RoundRobin': '#9C27B0',
}


# ═══════════════════════════════════════════════════════════════════════════
# Training Log I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_training_log(log: Dict, filepath: str):
    """Save training metrics as JSON."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # Convert numpy types to Python native for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert(log), f, indent=2, ensure_ascii=False)
    print(f"  Log saved: {filepath}")


def load_training_log(filepath: str) -> Dict:
    """Load training metrics from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_logs(log_dir: str) -> List[str]:
    """Find all training log JSON files in a directory."""
    pattern = os.path.join(log_dir, 'train_*.json')
    return sorted(glob.glob(pattern))


# ═══════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    log: Dict,
    save_path: Optional[str] = None,
    smooth_window: int = 50,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plot 3-panel training curves: reward, makespan, epsilon.

    Args:
        log: Training log dict from train_dqn.py (or load_training_log).
        save_path: If set, save figure to this path.
        smooth_window: Moving average window size.
        figsize: Figure dimensions.
    """
    problem = log.get('problem', '?')
    cfg = log.get('config', {})
    episodes = np.array(log['episodes'])
    rewards = np.array(log['rewards'])
    makespans = np.array(log['makespans'])
    best_ms = log.get('best_makespan', float('inf'))
    baselines = log.get('baselines', {})

    # Compute moving averages
    def moving_avg(x, w):
        if len(x) < w:
            return np.full_like(x, np.nan)
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode='valid')

    rew_ma = moving_avg(rewards, smooth_window)
    ms_ma = moving_avg(makespans, smooth_window)

    # Compute epsilon curve (reconstruct from linear decay)
    eps_decay = cfg.get('epsilon_decay', 100000)
    eps_start, eps_end = 1.0, 0.02
    # Estimate steps per episode from total steps
    est_steps = np.cumsum(np.full_like(episodes, 80))  # rough estimate
    epsilons = np.maximum(eps_end, eps_start +
                          (eps_end - eps_start) * np.minimum(est_steps / eps_decay, 1.0))

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    title = f"Training Curves — {problem}"
    if cfg:
        title += f"  (N-step={cfg.get('n_step','?')}, γ={cfg.get('gamma','?')})"
    fig.suptitle(title)

    # Panel 1: Reward
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.2, color='#2196F3', linewidth=0.5, label='Episode')
    if len(rew_ma) > 0:
        ax.plot(episodes[smooth_window-1:], rew_ma, color='#2196F3',
                linewidth=1.5, label=f'MA({smooth_window})')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Panel 2: Makespan
    ax = axes[1]
    ax.plot(episodes, makespans, alpha=0.2, color='#FF5722', linewidth=0.5, label='Episode')
    if len(ms_ma) > 0:
        ax.plot(episodes[smooth_window-1:], ms_ma, color='#FF5722',
                linewidth=1.5, label=f'MA({smooth_window})')
    ax.axhline(y=best_ms, color='red', linestyle='--', linewidth=1,
               label=f'Best={best_ms:.0f}')
    # Baseline lines
    for key, label, color in [('greedy_makespan', 'Greedy', '#4CAF50'),
                               ('roundrobin_makespan', 'RoundRobin', '#9C27B0'),
                               ('random_makespan', 'Random', '#FF9800')]:
        if key in baselines:
            ax.axhline(y=baselines[key], color=color, linestyle=':', linewidth=1,
                       label=f'{label}={baselines[key]:.0f}')
    ax.set_ylabel('Makespan')
    ax.legend(loc='upper right', ncol=2)

    # Panel 3: Epsilon
    ax = axes[2]
    ax.plot(episodes, epsilons, color='#4CAF50', linewidth=1.5)
    ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=1,
               label='Exploitation threshold (ε=0.3)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_method_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Grouped bar chart comparing methods.

    Args:
        results: {method_name: {'makespan': float, 'fatigue': float}}
        title: Chart title.
        save_path: Optional save path.
        figsize: Figure dimensions.
    """
    methods = list(results.keys())
    makespans = [results[m].get('makespan', results[m].get('makespan_mean', 0))
                 for m in methods]
    fatigues = [results[m].get('fatigue', results[m].get('fatigue_mean', 0))
                for m in methods]

    # Add error bars if available
    ms_errs = [results[m].get('makespan_std', 0) for m in methods]
    fat_errs = [results[m].get('fatigue_std', 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    x = np.arange(len(methods))
    bar_colors = [COLORS.get(m.split()[0], '#607D8B') for m in methods]

    # Makespan
    bars1 = ax1.bar(x, makespans, color=bar_colors, edgecolor='white',
                    yerr=ms_errs if any(ms_errs) else None, capsize=5)
    ax1.set_ylabel('Makespan')
    ax1.set_title('Makespan (lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    for bar, val in zip(bars1, makespans):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(makespans) * 0.01,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # Fatigue
    bars2 = ax2.bar(x, fatigues, color=bar_colors, edgecolor='white',
                    yerr=fat_errs if any(fat_errs) else None, capsize=5)
    ax2.set_ylabel('Avg Fatigue')
    ax2.set_title('Worker Fatigue (lower is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right')
    for bar, val in zip(bars2, fatigues):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(fatigues) * 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_dataset_comparison(
    all_dataset_results: Dict[str, Dict[str, Dict]],
    metric: str = 'makespan',
    save_path: Optional[str] = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Grouped bar chart across datasets.

    Args:
        all_dataset_results: {dataset_name: {method_name: {metric: value}}}
        metric: 'makespan' or 'fatigue'
        save_path: Optional save path.
        figsize: Figure dimensions.
    """
    datasets = list(all_dataset_results.keys())
    # Find all methods across datasets
    all_methods = set()
    for ds_results in all_dataset_results.values():
        all_methods.update(ds_results.keys())
    methods = sorted(all_methods)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(datasets))
    n_methods = len(methods)
    bar_width = 0.8 / n_methods

    for i, method in enumerate(methods):
        values = []
        for ds in datasets:
            if method in all_dataset_results[ds]:
                m = all_dataset_results[ds][method]
                values.append(m.get(metric, m.get(f'{metric}_mean', 0)))
            else:
                values.append(0)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = COLORS.get(method.split()[0], f'C{i}')
        bars = ax.bar(x + offset, values, bar_width, label=method,
                      color=color, edgecolor='white')

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left')

    ylabel = 'Makespan' if metric == 'makespan' else 'Avg Fatigue'
    ax.set_ylabel(ylabel)
    if metric == 'makespan':
        ax.set_title('Makespan Comparison Across Datasets (lower is better)')
    else:
        ax.set_title('Fatigue Comparison Across Datasets (lower is better)')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_nstep_ablation(
    n_step_results: Dict[int, Dict[str, float]],
    baseline_ms: float = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Line chart showing N-step sensitivity.

    Args:
        n_step_results: {n: {'makespan_mean': float, 'makespan_std': float}}
        baseline_ms: Greedy baseline makespan for reference line.
        save_path: Optional save path.
        figsize: Figure dimensions.
    """
    n_values = sorted(n_step_results.keys())
    ms_means = [n_step_results[n].get('makespan_mean',
                                       n_step_results[n].get('makespan', 0))
                for n in n_values]
    ms_stds = [n_step_results[n].get('makespan_std', 0) for n in n_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('N-Step Return Ablation Study')

    # Makespan
    ax1.errorbar(n_values, ms_means, yerr=ms_stds, marker='o', linewidth=2,
                 markersize=8, capsize=5, color='#2196F3')
    if baseline_ms is not None:
        ax1.axhline(y=baseline_ms, color='#4CAF50', linestyle='--', linewidth=2,
                    label=f'Greedy Baseline ({baseline_ms:.0f})')
    ax1.set_xlabel('N-step')
    ax1.set_ylabel('Makespan')
    ax1.set_title('Makespan vs N-step')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Fatigue
    fat_means = [n_step_results[n].get('fatigue_mean',
                                        n_step_results[n].get('fatigue', 0))
                 for n in n_values]
    fat_stds = [n_step_results[n].get('fatigue_std', 0) for n in n_values]
    ax2.errorbar(n_values, fat_means, yerr=fat_stds, marker='s', linewidth=2,
                 markersize=8, capsize=5, color='#FF5722')
    ax2.set_xlabel('N-step')
    ax2.set_ylabel('Avg Fatigue')
    ax2.set_title('Fatigue vs N-step')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_fatigue_comparison(
    fatigue_results: Dict[float, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Dual-axis plot: λ vs makespan and fatigue.

    Args:
        fatigue_results: {lambda_val: {'makespan_mean': float, 'fatigue_mean': float}}
        save_path: Optional save path.
        figsize: Figure dimensions.
    """
    lambdas = sorted(fatigue_results.keys())
    ms_vals = [fatigue_results[lam].get('makespan_mean',
                                         fatigue_results[lam].get('makespan', 0))
               for lam in lambdas]
    fat_vals = [fatigue_results[lam].get('fatigue_mean',
                                          fatigue_results[lam].get('fatigue', 0))
                for lam in lambdas]

    fig, ax1 = plt.subplots(figsize=figsize)

    color_ms = '#2196F3'
    color_fat = '#FF5722'

    ax1.set_xlabel('λ (Fatigue Penalty Weight)')
    ax1.set_ylabel('Makespan', color=color_ms)
    line1 = ax1.plot(lambdas, ms_vals, marker='o', linewidth=2, markersize=8,
                     color=color_ms, label='Makespan')
    ax1.tick_params(axis='y', labelcolor=color_ms)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Fatigue', color=color_fat)
    line2 = ax2.plot(lambdas, fat_vals, marker='s', linewidth=2, markersize=8,
                     color=color_fat, label='Fatigue')
    ax2.tick_params(axis='y', labelcolor=color_fat)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')

    plt.title('Fatigue Penalty Weight Sensitivity (λ)')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


def plot_multi_run_comparison(
    logs: List[Dict],
    labels: List[str],
    save_path: Optional[str] = None,
    smooth_window: int = 50,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Compare training runs by plotting makespan curves together.

    Args:
        logs: List of training log dicts.
        labels: Labels for each log (e.g. ['N=1', 'N=3', 'N=5']).
        save_path: Optional save path.
        smooth_window: Moving average window.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for log, label in zip(logs, labels):
        makespans = np.array(log['makespans'])
        episodes = np.array(log['episodes'])

        if len(makespans) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            ma = np.convolve(makespans, kernel, mode='valid')
            ax.plot(episodes[smooth_window-1:], ma, linewidth=1.5, label=label)
        else:
            ax.plot(episodes, makespans, alpha=0.4, linewidth=0.8, label=label)

        # Mark best
        best_ms = log.get('best_makespan', float('inf'))
        ax.axhline(y=best_ms, linestyle=':', linewidth=0.8, alpha=0.5)

    # Baseline
    baselines = logs[0].get('baselines', {})
    if 'greedy_makespan' in baselines:
        ax.axhline(y=baselines['greedy_makespan'], color='#4CAF50',
                   linestyle='--', linewidth=1.5, label='Greedy SPT')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Makespan (moving avg)')
    ax.set_title('Multi-Run Comparison')
    ax.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Summary Tables
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results: Dict[str, Dict[str, Dict]], title: str = "Summary"):
    """Print a formatted comparison table to stdout.

    Args:
        all_results: {dataset: {method: {'makespan': ..., 'fatigue': ...}}}
        title: Table title.
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Header
    methods = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))
    col_width = max(12, max(len(m) for m in methods))

    header = f"{'Dataset':<10}"
    for m in methods:
        header += f" {m:>{col_width}}"
    print(header)
    print("-" * len(header))

    for ds, res in all_results.items():
        row = f"{ds:<10}"
        for m in methods:
            if m in res:
                ms = res[m].get('makespan', res[m].get('makespan_mean', 0))
                row += f" {ms:>{col_width}.1f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print("-" * len(header))

    # Also print fatigue table
    print(f"\n  Fatigue:")
    header = f"{'Dataset':<10}"
    for m in methods:
        header += f" {m:>{col_width}}"
    print(header)
    print("-" * len(header))

    for ds, res in all_results.items():
        row = f"{ds:<10}"
        for m in methods:
            if m in res:
                fat = res[m].get('fatigue', res[m].get('fatigue_mean', 0))
                row += f" {fat:>{col_width}.3f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)

    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="JSP Scheduling — Visualization & Comparison"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to a single training log JSON to plot training curves."
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Directory of training logs to compare multiple runs."
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save generated charts (default: ./charts/)."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display charts (useful for headless/scripted runs)."
    )
    args = parser.parse_args()

    set_style()
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'charts'
    )

    if args.log:
        # Single training log — plot training curves
        log = load_training_log(args.log)
        problem = log.get('problem', 'unknown')
        save_path = os.path.join(save_dir, f'training_curves_{problem}.png')
        plot_training_curves(log, save_path=save_path)

        # Also show method comparison from baselines + final eval
        baselines = log.get('baselines', {})
        final_eval = log.get('final_eval', {})
        results = {}
        if 'greedy_makespan' in baselines:
            results['Greedy SPT'] = {
                'makespan': baselines['greedy_makespan'],
                'fatigue': baselines['greedy_fatigue'],
            }
        if 'roundrobin_makespan' in baselines:
            results['Round-Robin'] = {
                'makespan': baselines['roundrobin_makespan'],
                'fatigue': baselines['roundrobin_fatigue'],
            }
        if 'random_makespan' in baselines:
            results['Random'] = {
                'makespan': baselines['random_makespan'],
                'fatigue': baselines['random_fatigue'],
            }
        if 'avg_makespan' in final_eval:
            results['DQN'] = {
                'makespan': final_eval['avg_makespan'],
                'fatigue': final_eval['avg_fatigue'],
                'makespan_std': final_eval.get('std_makespan', 0),
                'fatigue_std': final_eval.get('std_fatigue', 0),
            }
        if results:
            save_path = os.path.join(save_dir, f'method_comparison_{problem}.png')
            plot_method_comparison(results, title=f"Method Comparison — {problem}",
                                   save_path=save_path)

    elif args.compare:
        # Compare multiple logs (e.g., different N-step values)
        log_files = find_logs(args.compare)
        if not log_files:
            print(f"No training logs found in: {args.compare}")
            return

        logs = []
        labels = []
        for lf in log_files:
            log = load_training_log(lf)
            logs.append(log)
            cfg = log.get('config', {})
            n_step = cfg.get('n_step', '?')
            gamma = cfg.get('gamma', '?')
            labels.append(f"N={n_step}, γ={gamma} ({log.get('problem', '?')})")

        save_path = os.path.join(save_dir, 'multi_run_comparison.png')
        plot_multi_run_comparison(logs, labels, save_path=save_path)

    else:
        # Default: scan logs/ and generate charts for all found logs
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        log_files = find_logs(log_dir)

        if not log_files:
            print(f"No training logs found in {log_dir}.")
            print("Run train_dqn.py first to generate logs, then re-run visualize.py.")
            print("\nAlternatively, provide --log or --compare to specify log locations.")
            return

        print(f"Found {len(log_files)} training log(s).")

        for lf in log_files:
            print(f"\nProcessing: {os.path.basename(lf)}")
            log = load_training_log(lf)
            problem = log.get('problem', 'unknown')

            # Training curves
            save_path = os.path.join(save_dir, f'training_curves_{problem}.png')
            plot_training_curves(log, save_path=save_path)

            # Method comparison
            baselines = log.get('baselines', {})
            final_eval = log.get('final_eval', {})
            results = {}
            if 'greedy_makespan' in baselines:
                results['Greedy SPT'] = {
                    'makespan': baselines['greedy_makespan'],
                    'fatigue': baselines['greedy_fatigue'],
                }
            if 'roundrobin_makespan' in baselines:
                results['Round-Robin'] = {
                    'makespan': baselines['roundrobin_makespan'],
                    'fatigue': baselines['roundrobin_fatigue'],
                }
            if 'random_makespan' in baselines:
                results['Random'] = {
                    'makespan': baselines['random_makespan'],
                    'fatigue': baselines['random_fatigue'],
                }
            if 'avg_makespan' in final_eval:
                results['DQN'] = {
                    'makespan': final_eval['avg_makespan'],
                    'fatigue': final_eval['avg_fatigue'],
                    'makespan_std': final_eval.get('std_makespan', 0),
                    'fatigue_std': final_eval.get('std_fatigue', 0),
                }
            if results:
                save_path = os.path.join(save_dir, f'method_comparison_{problem}.png')
                plot_method_comparison(results,
                                       title=f"Method Comparison — {problem}",
                                       save_path=save_path)

            # Print text summary
            if results:
                print_summary_table({problem: results}, title=f"Results — {problem}")

    if not args.no_show:
        print("\nDone. Charts saved to:", save_dir)


if __name__ == "__main__":
    main()
