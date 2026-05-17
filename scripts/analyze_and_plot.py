#!/usr/bin/env python3
"""
Statistical analysis + visualization for LE SSERAFIM thigh measurement paper
Run after smpl_fitter_v3.py completes
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Load results
with open("/mnt/d/HERMES/study/lesserafim-thigh-paper/output/measurements_smpl_v3/results.json") as f:
    data = json.load(f)

print(f"Total measurements: {len(data)}")

# Group by member × era
members = ['sakura', 'chaewon', 'yunjin', 'kazuha', 'eunchae']
eras = ['antifragile', 'unforgiven', 'easy', 'crazy', 'hot']

# Member colors
colors = {'sakura': '#FF6B8A', 'chaewon': '#7EB8DA', 'yunjin': '#98D8C8',
          'kazuha': '#FFD700', 'eunchae': '#DDA0DD'}

# 1. Summary statistics
print("\n=== OVERALL STATISTICS ===")
table_data = []
for m in members:
    vals = [r['circ_cm'] for r in data if r['member'] == m]
    if vals:
        mean = np.mean(vals)
        std = np.std(vals)
        n = len(vals)
        print(f"{m:10s}: {mean:5.1f} ± {std:4.1f} cm (n={n:3d})")
        table_data.append({'member': m, 'mean': mean, 'std': std, 'n': n})

# 2. ANOVA: Is there a significant difference between members?
groups_by_member = [[r['circ_cm'] for r in data if r['member'] == m] for m in members]
groups_by_member = [g for g in groups_by_member if len(g) > 1]

if len(groups_by_member) >= 2:
    f_stat, p_val = stats.f_oneway(*groups_by_member)
    print(f"\n=== ANOVA (between members) ===")
    print(f"F = {f_stat:.3f}, p = {p_val:.6f}")
    print(f"Significant: {'YES' if p_val < 0.05 else 'NO'}")

# 3. Per-era per-member matrix
print("\n=== MEMBER × ERA MATRIX (mean cm) ===")
header = f"{'':>10}" + "".join(f"{e:>14}" for e in eras)
print(header)
for m in members:
    row = f"{m:>10}"
    for e in eras:
        vals = [r['circ_cm'] for r in data if r['member'] == m and r['era'] == e]
        if vals:
            row += f"{np.mean(vals):>8.1f} (n={len(vals):2d})"
        else:
            row += f"{'N/A':>14}"
    print(row)

# 4. Longitudinal analysis per member
print("\n=== LONGITUDINAL (era order) ===")
for m in members:
    eras_with_data = [(e, [r['circ_cm'] for r in data if r['member'] == m and r['era'] == e])
                      for e in eras]
    vals_by_era = [(e, v) for e, v in eras_with_data if v]
    if len(vals_by_era) >= 2:
        means = [np.mean(v) for _, v in vals_by_era]
        era_labels = [e for e, _ in vals_by_era]
        print(f"{m}: {' → '.join(f'{mv:.1f}cm' for mv in means)} ({', '.join(era_labels)})")

# 5. Generate charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LE SSERAFIM Thigh Circumference Analysis\n(SMPL Body Model + Height Calibration)', fontsize=14)

# (a) Bar chart by member
ax = axes[0, 0]
means = [np.mean([r['circ_cm'] for r in data if r['member'] == m]) for m in members if [r for r in data if r['member'] == m]]
stds = [np.std([r['circ_cm'] for r in data if r['member'] == m]) for m in members if [r for r in data if r['member'] == m]]
present_members = [m for m in members if [r for r in data if r['member'] == m]]
bars = ax.bar(present_members, means, yerr=stds, capsize=5,
             color=[colors[m] for m in present_members], alpha=0.8)
ax.set_ylabel('Thigh Circumference (cm)')
ax.set_title('Overall by Member')
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

# (b) Era trend per member
ax = axes[0, 1]
for m in members:
    era_means = []
    era_labels = []
    for e in eras:
        vals = [r['circ_cm'] for r in data if r['member'] == m and r['era'] == e]
        if len(vals) >= 2:
            era_means.append(np.mean(vals))
            era_labels.append(e[:4])
    if era_means:
        ax.plot(range(len(era_means)), era_means, 'o-', label=m, color=colors[m], linewidth=2)
ax.set_xticks(range(len(eras)))
ax.set_xticklabels([e[:4] for e in eras])
ax.set_ylabel('Thigh Circumference (cm)')
ax.set_title('Era Trend (min 2 samples)')
ax.legend(loc='best', fontsize=8)

# (c) Distribution violin plot
ax = axes[1, 0]
all_vals = [[r['circ_cm'] for r in data if r['member'] == m] for m in present_members]
parts = ax.violinplot(all_vals, positions=range(len(present_members)), showmeans=True)
if parts:
    ax.set_xticks(range(len(present_members)))
    ax.set_xticklabels(present_members)
ax.set_ylabel('Thigh Circumference (cm)')
ax.set_title('Distribution by Member')

# (d) Sample sizes
ax = axes[1, 1]
sample_sizes = [len([r for r in data if r['member'] == m]) for m in present_members]
ax.bar(present_members, sample_sizes, color=[colors[m] for m in present_members], alpha=0.7)
ax.set_ylabel('Number of Measurements')
ax.set_title('Sample Size by Member')

plt.tight_layout()
fig.savefig("/mnt/d/HERMES/study/lesserafim-thigh-paper/output/figures/analysis.png", dpi=150)
print(f"\nChart saved to output/figures/analysis.png")

# Save summary JSON
summary = {
    'members': {m: {
        'mean_cm': round(np.mean([r['circ_cm'] for r in data if r['member'] == m]), 1),
        'std_cm': round(np.std([r['circ_cm'] for r in data if r['member'] == m]), 1),
        'n': len([r for r in data if r['member'] == m]),
        'by_era': {e: {
            'mean_cm': round(np.mean([r['circ_cm'] for r in data if r['member'] == m and r['era'] == e]), 1),
            'n': len([r for r in data if r['member'] == m and r['era'] == e])
        } for e in eras if [r for r in data if r['member'] == m and r['era'] == e]}
    } for m in members if [r for r in data if r['member'] == m]}
}
with open("/mnt/d/HERMES/study/lesserafim-thigh-paper/output/measurements_smpl_v3/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved!")
