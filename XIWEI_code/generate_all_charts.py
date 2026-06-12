"""
Supplementary script to generate chart types 3, 4, 5 from visualize.py.
- Chart 3: Dataset Comparison (cross-dataset grouped bar chart)
- Chart 4: N-step Ablation
- Chart 5: Fatigue Comparison (lambda sensitivity)
- Also: Per-problem Multi-run Comparison

Run from XIWEI_code/ directory: python generate_all_charts.py
"""

import os
import sys
import json
import glob
import numpy as np

# Ensure we can import visualize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize import (
    set_style,
    load_training_log,
    find_logs,
    _get_model_label,
    plot_dataset_comparison,
    plot_nstep_ablation,
    plot_fatigue_comparison,
    plot_multi_run_comparison,
    COLORS,
)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
os.makedirs(SAVE_DIR, exist_ok=True)
set_style()


def build_dataset_comparison_data(all_log_files):
    """Extract the best makespan & fatigue for each (dataset, method) pair.

    For RL methods, use final_eval avg values.
    For baselines, use the baseline values from any log for that problem.
    """
    # Collect data: {problem: {method: {'makespan': ..., 'fatigue': ...}}}
    results = {}

    for lf in all_log_files:
        log = load_training_log(lf)
        problem = log.get('problem', '?')
        if problem == '?':
            continue

        if problem not in results:
            results[problem] = {}

        # Baselines (take first occurrence — they're deterministic per problem)
        baselines = log.get('baselines', {})
        if 'Greedy SPT' not in results[problem] and 'greedy_makespan' in baselines:
            results[problem]['Greedy SPT'] = {
                'makespan': baselines['greedy_makespan'],
                'fatigue': baselines['greedy_fatigue'],
            }
        if 'Round-Robin' not in results[problem] and 'roundrobin_makespan' in baselines:
            results[problem]['Round-Robin'] = {
                'makespan': baselines['roundrobin_makespan'],
                'fatigue': baselines['roundrobin_fatigue'],
            }
        if 'Random' not in results[problem] and 'random_makespan' in baselines:
            results[problem]['Random'] = {
                'makespan': baselines['random_makespan'],
                'fatigue': baselines['random_fatigue'],
            }

        # RL method
        final_eval = log.get('final_eval', {})
        if 'avg_makespan' in final_eval or 'makespan_mean' in final_eval:
            label = _get_model_label(log)
            ms = final_eval.get('avg_makespan', final_eval.get('makespan_mean', 0))
            fat = final_eval.get('avg_fatigue', final_eval.get('fatigue_mean', 0))
            ms_std = final_eval.get('std_makespan', final_eval.get('makespan_std', 0))
            fat_std = final_eval.get('std_fatigue', final_eval.get('fatigue_std', 0))

            # Keep the best (lowest makespan) if multiple runs for same method+problem
            if label not in results[problem] or ms < results[problem][label]['makespan']:
                results[problem][label] = {
                    'makespan': ms,
                    'fatigue': fat,
                    'makespan_std': ms_std,
                    'fatigue_std': fat_std,
                }

    return results


def build_nstep_data(all_log_files):
    """Build N-step ablation data.

    With current logs, we have DQN (n_step=3) and DQN_V2 (n_step=7) but
    different architectures. We compare them per problem where both exist.
    """
    # Collect: {problem: {n_step: {'makespan_mean': ..., 'makespan_std': ...}}}
    by_problem = {}

    for lf in all_log_files:
        log = load_training_log(lf)
        problem = log.get('problem', '?')
        cfg = log.get('config', {})
        n_step = cfg.get('n_step', None)
        if n_step is None:
            continue

        final_eval = log.get('final_eval', {})
        if 'avg_makespan' not in final_eval and 'makespan_mean' not in final_eval:
            continue

        if problem not in by_problem:
            by_problem[problem] = {}

        ms_key = final_eval.get('avg_makespan', final_eval.get('makespan_mean', 0))
        fat_key = final_eval.get('avg_fatigue', final_eval.get('fatigue_mean', 0))
        ms_std = final_eval.get('std_makespan', final_eval.get('makespan_std', 0))
        fat_std = final_eval.get('std_fatigue', final_eval.get('fatigue_std', 0))

        n_step_int = int(n_step)
        method = _get_model_label(log)
        label = f"{method} (N={n_step_int})"

        by_problem[problem][n_step_int] = {
            'makespan_mean': float(ms_key),
            'makespan_std': float(ms_std),
            'fatigue_mean': float(fat_key),
            'fatigue_std': float(fat_std),
            'label': label,
        }

    return by_problem


def build_fatigue_lambda_data(all_log_files):
    """Build fatigue lambda sensitivity data.

    Most logs use lambda_fatigue=2.0. We extract whatever variation exists.
    """
    by_problem = {}

    for lf in all_log_files:
        log = load_training_log(lf)
        problem = log.get('problem', '?')
        cfg = log.get('config', {})
        lam = cfg.get('lambda_fatigue', None)
        if lam is None:
            continue

        final_eval = log.get('final_eval', {})
        if 'avg_makespan' not in final_eval and 'makespan_mean' not in final_eval:
            continue

        if problem not in by_problem:
            by_problem[problem] = {}

        lam_float = float(lam)
        ms_key = final_eval.get('avg_makespan', final_eval.get('makespan_mean', 0))
        fat_key = final_eval.get('avg_fatigue', final_eval.get('fatigue_mean', 0))

        by_problem[problem][lam_float] = {
            'makespan_mean': float(ms_key),
            'fatigue_mean': float(fat_key),
        }

    return by_problem


def main():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    all_log_files = find_logs(log_dir)

    if not all_log_files:
        print("No log files found!")
        return

    print(f"Processing {len(all_log_files)} log files...")

    # ── Chart 3: Dataset Comparison ──
    print("\n" + "=" * 60)
    print("Chart 3: Dataset Comparison")
    print("=" * 60)
    dataset_data = build_dataset_comparison_data(all_log_files)

    # Sort datasets by size
    def sort_key(name):
        parts = name.split('x')
        return tuple(int(p) for p in parts)
    sorted_datasets = sorted(dataset_data.keys(), key=sort_key)

    ordered_data = {ds: dataset_data[ds] for ds in sorted_datasets}
    print(f"  Datasets: {sorted_datasets}")
    for ds in sorted_datasets:
        print(f"    {ds}: {list(ordered_data[ds].keys())}")

    plot_dataset_comparison(
        ordered_data,
        metric='makespan',
        save_path=os.path.join(SAVE_DIR, 'dataset_comparison_makespan.png'),
    )
    plot_dataset_comparison(
        ordered_data,
        metric='fatigue',
        save_path=os.path.join(SAVE_DIR, 'dataset_comparison_fatigue.png'),
    )

    # ── Chart 4: N-step Ablation ──
    print("\n" + "=" * 60)
    print("Chart 4: N-step Ablation")
    print("=" * 60)
    nstep_data = build_nstep_data(all_log_files)
    for problem, data in nstep_data.items():
        print(f"  {problem}: n_steps={list(data.keys())}")
        if len(data) >= 2:
            # Get greedy baseline for this problem
            baseline_ms = None
            for lf in all_log_files:
                log = load_training_log(lf)
                if log.get('problem') == problem:
                    bl = log.get('baselines', {})
                    if 'greedy_makespan' in bl:
                        baseline_ms = bl['greedy_makespan']
                        break

            plot_nstep_ablation(
                data,
                baseline_ms=baseline_ms,
                save_path=os.path.join(SAVE_DIR, f'nstep_ablation_{problem}.png'),
            )

    # ── Chart 5: Fatigue Lambda Comparison ──
    print("\n" + "=" * 60)
    print("Chart 5: Fatigue Lambda Comparison")
    print("=" * 60)
    lambda_data = build_fatigue_lambda_data(all_log_files)
    for problem, data in lambda_data.items():
        print(f"  {problem}: lambdas={list(data.keys())}")
        if len(data) >= 2:
            plot_fatigue_comparison(
                data,
                save_path=os.path.join(SAVE_DIR, f'fatigue_lambda_{problem}.png'),
            )
    if all(len(d) < 2 for d in lambda_data.values()):
        print("  WARNING: Only one lambda value (2.0) found across all logs.")
        print("  Cannot generate meaningful fatigue penalty sensitivity chart.")
        print("  To generate this chart, train models with different lambda_fatigue values.")

    # ── Bonus: Per-Problem Multi-Run Comparison ──
    print("\n" + "=" * 60)
    print("Bonus: Per-Problem Multi-Run Comparison")
    print("=" * 60)

    # Group logs by problem
    logs_by_problem = {}
    for lf in all_log_files:
        log = load_training_log(lf)
        problem = log.get('problem', '?')
        if problem not in logs_by_problem:
            logs_by_problem[problem] = []
        logs_by_problem[problem].append((lf, log))

    for problem, entries in sorted(logs_by_problem.items(), key=lambda x: sort_key(x[0])):
        if len(entries) < 2:
            continue
        print(f"  {problem}: {len(entries)} runs")

        logs_list = []
        labels_list = []
        for lf, log in entries:
            logs_list.append(log)
            cfg = log.get('config', {})
            method = _get_model_label(log)
            n_step = cfg.get('n_step', '?')
            labels_list.append(f"{method} (N={n_step})")

        plot_multi_run_comparison(
            logs_list,
            labels_list,
            save_path=os.path.join(SAVE_DIR, f'multi_run_{problem}.png'),
        )

    print(f"\nDone! All supplementary charts saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
