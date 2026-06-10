"""
Analyze DQN V2 results vs Original DQN, PPO, and baselines.
Generates comparison charts (with _v2 suffix to avoid overwriting).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize import set_style, COLORS

set_style()

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Load all results ──────────────────────────────────────────────────────
def load_dqn_v2_results():
    """Load DQN V2 300-ep results from combined JSON."""
    path = os.path.join(LOG_DIR, "dqn_v2_combined_results.json")
    with open(path, 'r') as f:
        return json.load(f)

def load_log(log_name):
    """Load a training log JSON."""
    path = os.path.join(LOG_DIR, log_name)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

# ─── Main analysis ─────────────────────────────────────────────────────────
def main():
    v2 = load_dqn_v2_results()

    # Collect all data
    datasets = ["6x6x2", "10x10x3", "15x10x3", "20x10x3", "30x10x3",
                "10x5x3", "10x10x6", "15x10x2", "15x5x3", "20x5x3"]

    # Short labels for charts
    short_labels = [d for d in datasets]

    data = {}  # ds -> {method: {makespan, fatigue, best_ms}}

    for ds in datasets:
        data[ds] = {}

        # DQN V2 (300ep)
        if ds in v2 and "300" in v2[ds]:
            v2_data = v2[ds]["300"]
            data[ds]["DQN_V2"] = {
                "makespan": v2_data["makespan_mean"],
                "best_ms": v2_data["best_makespan"],
                "fatigue": v2_data["fatigue_mean"],
            }

        # Original DQN (300ep)
        dqn_log = load_log(f"train_{ds}_dqn.json")
        if dqn_log:
            eval_data = dqn_log.get("final_eval", {})
            data[ds]["DQN"] = {
                "makespan": eval_data.get("makespan_mean", dqn_log.get("best_makespan", 0)),
                "best_ms": dqn_log.get("best_makespan", 0),
                "fatigue": eval_data.get("fatigue_mean", dqn_log.get("best_fatigue", 0)),
            }

        # PPO (300ep)
        ppo_log = load_log(f"train_{ds}_ppo.json")
        if ppo_log:
            eval_data = ppo_log.get("final_eval", {})
            data[ds]["PPO"] = {
                "makespan": eval_data.get("makespan_mean", ppo_log.get("best_makespan", 0)),
                "best_ms": ppo_log.get("best_makespan", 0),
                "fatigue": eval_data.get("fatigue_mean", ppo_log.get("best_fatigue", 0)),
            }

        # Baselines from any log
        log = dqn_log or ppo_log
        if log and "baselines" in log:
            bl = log["baselines"]
            data[ds]["Greedy"] = {"makespan": bl["greedy_makespan"], "fatigue": bl["greedy_fatigue"]}
            data[ds]["Random"] = {"makespan": bl.get("random_makespan", 0), "fatigue": bl.get("random_fatigue", 0)}
            data[ds]["RoundRobin"] = {"makespan": bl.get("roundrobin_makespan", 0), "fatigue": bl.get("roundrobin_fatigue", 0)}

    # ── Print analysis table ────────────────────────────────────────────────
    print("=" * 80)
    print("DQN V2 Optimization Analysis — 300-Episode Results")
    print("=" * 80)

    methods = ["Greedy", "Random", "RoundRobin", "DQN", "DQN_V2", "PPO"]

    print(f"\n{'Dataset':<12}", end="")
    for m in methods:
        print(f" {m:>12}", end="")
    print(f" {'V2vsDQN':>9} {'V2vsGreedy':>10}")
    print("-" * (12 + 13*len(methods) + 19))

    for ds in datasets:
        if ds not in data:
            continue
        print(f"{ds:<12}", end="")
        greedy_ms = data[ds].get("Greedy", {}).get("makespan", 0)
        dqn_ms = data[ds].get("DQN", {}).get("makespan", 0)
        v2_ms = data[ds].get("DQN_V2", {}).get("makespan", 0)

        for m in methods:
            if m in data[ds]:
                val = data[ds][m]["makespan"]
                print(f" {val:>12.1f}", end="")
            else:
                print(f" {'N/A':>12}", end="")

        # V2 vs DQN and V2 vs Greedy deltas
        if dqn_ms > 0 and v2_ms > 0:
            v2_dqn_delta = (v2_ms - dqn_ms) / dqn_ms * 100
            print(f" {v2_dqn_delta:>+8.1f}%", end="")
        else:
            print(f" {'N/A':>9}", end="")

        if greedy_ms > 0 and v2_ms > 0:
            v2_greedy_delta = (v2_ms - greedy_ms) / greedy_ms * 100
            print(f" {v2_greedy_delta:>+9.1f}%", end="")
        else:
            print(f" {'N/A':>10}", end="")
        print()

    # ── Win/Loss summary ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("DQN V2 vs Original DQN — Per-Dataset Analysis")
    print(f"{'='*80}")

    wins = 0
    losses = 0
    for ds in datasets:
        if ds not in data or "DQN" not in data[ds] or "DQN_V2" not in data[ds]:
            continue
        dqn_ms = data[ds]["DQN"]["makespan"]
        v2_ms = data[ds]["DQN_V2"]["makespan"]
        delta = (v2_ms - dqn_ms) / dqn_ms * 100
        best_ms_v2 = data[ds]["DQN_V2"]["best_ms"]
        best_ms_dqn = data[ds]["DQN"]["best_ms"]
        greedy_ms = data[ds].get("Greedy", {}).get("makespan", 1)

        status = "[WIN]" if delta < 0 else "[LOSS]"
        if delta < 0:
            wins += 1
        else:
            losses += 1

        ds_label = f"{ds} (J={ds.split('x')[0]}, M={ds.split('x')[1]}, W={ds.split('x')[2]})"
        print(f"\n  {status} {ds_label}")
        print(f"    Greedy:          {greedy_ms:.1f}")
        print(f"    DQN Original:    {dqn_ms:.1f} (best: {best_ms_dqn:.1f})")
        print(f"    DQN V2:          {v2_ms:.1f} (best: {best_ms_v2:.1f})")
        print(f"    V2 vs Original:  {delta:+.1f}%")
        print(f"    V2 vs Greedy:    {(v2_ms-greedy_ms)/greedy_ms*100:+.1f}%")
        print(f"    Best V2 vs Best Original: {(best_ms_v2-best_ms_dqn)/best_ms_dqn*100:+.1f}%")

    print(f"\n  Total: {wins} wins, {losses} losses out of {wins+losses} datasets")

    # ── Key findings ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Key Findings")
    print(f"{'='*80}")

    # Compute average improvement on datasets where V2 wins
    win_deltas = []
    loss_deltas = []
    for ds in datasets:
        if ds not in data or "DQN" not in data[ds] or "DQN_V2" not in data[ds]:
            continue
        dqn_ms = data[ds]["DQN"]["makespan"]
        v2_ms = data[ds]["DQN_V2"]["makespan"]
        delta = (v2_ms - dqn_ms) / dqn_ms * 100
        if delta < 0:
            win_deltas.append(delta)
        else:
            loss_deltas.append(delta)

    if win_deltas:
        print(f"\n  On datasets where V2 WINS (n={len(win_deltas)}):")
        print(f"    Avg improvement: {np.mean(win_deltas):.1f}%")
        print(f"    Best improvement: {min(win_deltas):.1f}%")

    if loss_deltas:
        print(f"\n  On datasets where V2 LOSES (n={len(loss_deltas)}):")
        print(f"    Avg regression: {np.mean(loss_deltas):.1f}%")
        print(f"    Worst regression: {max(loss_deltas):.1f}%")

    # ── Generate charts ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Generating Charts")
    print(f"{'='*80}")

    v2_color = '#E91E63'  # Pink for V2
    orig_dqn_color = COLORS.get('DQN', '#2196F3')
    greedy_color = '#4CAF50'

    # Chart 1: Bar chart — DQN V2 vs Original DQN (Makespan) per dataset
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.25

    dqn_vals = [data[ds].get("DQN", {}).get("makespan", 0) for ds in datasets]
    v2_vals = [data[ds].get("DQN_V2", {}).get("makespan", 0) for ds in datasets]
    greedy_vals = [data[ds].get("Greedy", {}).get("makespan", 0) for ds in datasets]

    bars1 = ax.bar(x - width, greedy_vals, width, label='Greedy SPT', color=greedy_color, alpha=0.7)
    bars2 = ax.bar(x, dqn_vals, width, label='DQN Original (300ep)', color=orig_dqn_color, alpha=0.85)
    bars3 = ax.bar(x + width, v2_vals, width, label='DQN V2 (300ep)', color=v2_color, alpha=0.85)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Makespan')
    ax.set_title('DQN V2 vs Original DQN vs Greedy — Makespan Comparison (300 episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    # Add improvement labels
    for i, ds in enumerate(datasets):
        if dqn_vals[i] > 0 and v2_vals[i] > 0:
            delta = (v2_vals[i] - dqn_vals[i]) / dqn_vals[i] * 100
            color = 'green' if delta < 0 else 'red'
            ax.annotate(f'{delta:+.1f}%', (x[i] + width, v2_vals[i]),
                       textcoords="offset points", xytext=(0, 5),
                       ha='center', fontsize=7, color=color, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_vs_original_makespan.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_vs_original_makespan.png")

    # Chart 2: Improvement percentage bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    improvements = []
    for ds in datasets:
        dqn_ms = data[ds].get("DQN", {}).get("makespan", 0)
        v2_ms = data[ds].get("DQN_V2", {}).get("makespan", 0)
        if dqn_ms > 0:
            improvements.append((v2_ms - dqn_ms) / dqn_ms * 100)
        else:
            improvements.append(0)

    colors = ['#4CAF50' if imp < 0 else '#F44336' for imp in improvements]
    bars = ax.bar(datasets, improvements, color=colors, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Improvement over Original DQN (%)')
    ax.set_title('DQN V2 Improvement over Original DQN (300 episodes)')
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + (3 if imp >= 0 else -8),
               f'{imp:+.1f}%', ha='center', fontsize=9, fontweight='bold',
               color='green' if imp < 0 else 'red')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_improvement_pct.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_improvement_pct.png")

    # Chart 3: All methods comparison across datasets (Makespan)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.15

    method_configs = [
        ("Greedy", greedy_color, 0),
        ("Random", '#FF9800', 1),
        ("RoundRobin", '#9C27B0', 2),
        ("DQN", orig_dqn_color, 3),
        ("DQN_V2", v2_color, 4),
        ("PPO", COLORS.get('PPO', '#00BCD4'), 5),
    ]

    for name, color, offset in method_configs:
        vals = []
        for ds in datasets:
            if name in data[ds]:
                vals.append(data[ds][name]["makespan"])
            else:
                vals.append(0)
        bar = ax.bar(x + (offset - 2.5) * width, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Makespan')
    ax.set_title('All Methods Comparison — Makespan (300 episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_all_methods_comparison.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_all_methods_comparison.png")

    # Chart 4: Fatigue comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.2

    for name, color, offset in [("Greedy", greedy_color, -1), ("DQN", orig_dqn_color, 0),
                                 ("DQN_V2", v2_color, 1), ("PPO", COLORS.get('PPO', '#00BCD4'), 2)]:
        vals = []
        for ds in datasets:
            if name in data[ds]:
                vals.append(data[ds][name]["fatigue"])
            else:
                vals.append(0)
        ax.bar(x + offset * width, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Worker Fatigue')
    ax.set_title('Worker Fatigue Comparison (300 episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_fatigue_comparison.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_fatigue_comparison.png")

    # Chart 5: Best Makespan comparison (DQN vs DQN V2) — training best, not eval
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.3

    dqn_best = [data[ds].get("DQN", {}).get("best_ms", 0) for ds in datasets]
    v2_best = [data[ds].get("DQN_V2", {}).get("best_ms", 0) for ds in datasets]
    greedy_best = [data[ds].get("Greedy", {}).get("makespan", 0) for ds in datasets]

    ax.bar(x - width, greedy_best, width, label='Greedy SPT', color=greedy_color, alpha=0.7)
    ax.bar(x, dqn_best, width, label='DQN Original (best)', color=orig_dqn_color, alpha=0.85)
    ax.bar(x + width, v2_best, width, label='DQN V2 (best)', color=v2_color, alpha=0.85)

    # Add delta labels
    for i, ds in enumerate(datasets):
        if dqn_best[i] > 0 and v2_best[i] > 0:
            delta = (v2_best[i] - dqn_best[i]) / dqn_best[i] * 100
            color = 'green' if delta < 0 else 'red'
            ax.annotate(f'{delta:+.1f}%', (x[i] + width, v2_best[i]),
                       textcoords="offset points", xytext=(0, 5),
                       ha='center', fontsize=7, color=color, fontweight='bold')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Best Training Makespan')
    ax.set_title('Best Training Makespan: DQN V2 vs Original DQN (300 episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_best_makespan.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_best_makespan.png")

    # Chart 6: DQN V2 vs Greedy ratio (how much worse than optimal)
    fig, ax = plt.subplots(figsize=(10, 5))
    ratios_v2 = []
    ratios_dqn = []
    ratios_ppo = []
    valid_ds = []
    for ds in datasets:
        greedy_ms = data[ds].get("Greedy", {}).get("makespan", 0)
        if greedy_ms > 0:
            valid_ds.append(ds)
            v2_ms = data[ds].get("DQN_V2", {}).get("makespan", 0)
            dqn_ms = data[ds].get("DQN", {}).get("makespan", 0)
            ppo_ms = data[ds].get("PPO", {}).get("makespan", 0)
            ratios_v2.append(v2_ms / greedy_ms if v2_ms > 0 else 0)
            ratios_dqn.append(dqn_ms / greedy_ms if dqn_ms > 0 else 0)
            ratios_ppo.append(ppo_ms / greedy_ms if ppo_ms > 0 else 0)

    x = np.arange(len(valid_ds))
    width = 0.25
    ax.bar(x - width, ratios_dqn, width, label='DQN Original', color=orig_dqn_color, alpha=0.85)
    ax.bar(x, ratios_v2, width, label='DQN V2', color=v2_color, alpha=0.85)
    ax.bar(x + width, ratios_ppo, width, label='PPO', color=COLORS.get('PPO', '#00BCD4'), alpha=0.85)
    ax.axhline(y=1.0, color=greedy_color, linewidth=2, linestyle='--', label='Greedy SPT (=1.0)')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Makespan / Greedy Makespan')
    ax.set_title('RL Methods vs Greedy SPT (300 episodes, lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_ds, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(CHART_DIR, "dqn_v2_vs_greedy_ratio.png")
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"  Saved: charts/dqn_v2_vs_greedy_ratio.png")

    print(f"\n[OK] Analysis complete! 6 charts saved to charts/")
    print(f"  No existing charts were overwritten (all use _v2 suffix)")

    # ── Save analysis data as JSON for report ───────────────────────────────
    analysis = {
        "v2_wins": wins,
        "v2_losses": losses,
        "total_datasets": wins + losses,
        "win_deltas": [float(d) for d in win_deltas],
        "loss_deltas": [float(d) for d in loss_deltas],
        "avg_win_improvement": float(np.mean(win_deltas)) if win_deltas else 0,
        "avg_loss_regression": float(np.mean(loss_deltas)) if loss_deltas else 0,
        "per_dataset": {}
    }
    for ds in datasets:
        if ds in data and "DQN" in data[ds] and "DQN_V2" in data[ds]:
            dqn_ms = data[ds]["DQN"]["makespan"]
            v2_ms = data[ds]["DQN_V2"]["makespan"]
            greedy_ms = data[ds].get("Greedy", {}).get("makespan", 0)
            analysis["per_dataset"][ds] = {
                "greedy_ms": float(greedy_ms),
                "dqn_ms": float(dqn_ms),
                "v2_ms": float(v2_ms),
                "v2_vs_dqn_pct": float((v2_ms - dqn_ms) / dqn_ms * 100),
                "v2_vs_greedy_pct": float((v2_ms - greedy_ms) / greedy_ms * 100) if greedy_ms > 0 else 0,
            }

    with open(os.path.join(LOG_DIR, "dqn_v2_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis data saved: logs/dqn_v2_analysis.json")


if __name__ == "__main__":
    main()
