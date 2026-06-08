import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams['axes.unicode_minus'] = False

FT_ON  = 1.3490
FT_OFF = 1.3587
FR_ON  = 3.0967
FR_OFF = 5.0638

DROP_ON  = FR_ON  - FT_ON
DROP_OFF = FR_OFF - FT_OFF
REDUCTION_PCT = (DROP_OFF - DROP_ON) / DROP_OFF * 100

C_ON  = '#1565C0'
C_OFF = '#8B2500'
C_BG  = '#F0F4F8'

fig = plt.figure(figsize=(15, 7), facecolor=C_BG)
fig.suptitle('KernelNet Effect: Fine-tune vs Frozen Backbone',
             fontsize=16, fontweight='bold', color='#1A237E', y=1.00)

gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.06, right=0.97,
                      top=0.88, bottom=0.14)

# ── Panel 1: All 4 bars ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#FAFBFC')

labels = ['Ablation A\n(KNet+FT)', 'Baseline\n(No KNet+FT)',
          'Freeze Proposed\n(KNet+FR)', 'Freeze Baseline\n(No KNet+FR)']
vals   = [FT_ON, FT_OFF, FR_ON, FR_OFF]
colors = [C_ON, C_OFF, C_ON, C_OFF]
hatches = ['', '', '///', '///']

bars = ax1.bar(range(4), vals, color=colors, hatch=hatches,
               width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.08,
             f'{v:.4f}°', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#1A1A1A')

ax1.axvline(1.5, color='#B0BEC5', lw=1.5, linestyle='--', zorder=2)
ax1.text(0.5,  -0.62, 'Fine-tune', ha='center', fontsize=10,
         color='#546E7A', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#E8EAF6', alpha=0.8))
ax1.text(2.5,  -0.62, 'Frozen Backbone', ha='center', fontsize=10,
         color='#546E7A', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FCE4EC', alpha=0.8))

ax1.set_xticks(range(4))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Test Angular Error (deg)', fontsize=11)
ax1.set_title('All Conditions', fontsize=12, fontweight='bold', pad=8)
ax1.set_ylim(0, 6.5)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.spines[['top', 'right']].set_visible(False)

p1 = mpatches.Patch(color=C_ON,  label='KernelNet O')
p2 = mpatches.Patch(color=C_OFF, label='KernelNet X')
ax1.legend(handles=[p1, p2], fontsize=9, loc='upper left')

# ── Panel 2: Grouped by backbone condition ────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#FAFBFC')

x = np.array([0, 1.4])
w = 0.5
b_on  = ax2.bar(x - w/2, [FT_ON, FR_ON],  w, color=C_ON,  label='KernelNet O',
                edgecolor='white', lw=1.5, zorder=3)
b_off = ax2.bar(x + w/2, [FT_OFF, FR_OFF], w, color=C_OFF, label='KernelNet X',
                edgecolor='white', lw=1.5, zorder=3)

for bar, v in zip(list(b_on) + list(b_off), [FT_ON, FR_ON, FT_OFF, FR_OFF]):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.08,
             f'{v:.4f}°', ha='center', va='bottom',
             fontsize=10, fontweight='bold', color='#1A1A1A')

ax2.set_xticks(x)
ax2.set_xticklabels(['Fine-tune\n(Backbone trained)', 'Frozen\n(Backbone fixed)'], fontsize=10)
ax2.set_ylabel('Test Angular Error (deg)', fontsize=11)
ax2.set_title('KernelNet Effect by Backbone Condition', fontsize=12,
              fontweight='bold', pad=8)
ax2.set_ylim(0, 6.5)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines[['top', 'right']].set_visible(False)

# ── Panel 3: Performance drop when freezing ───────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('#FAFBFC')

bars3 = ax3.bar([0, 1], [DROP_ON, DROP_OFF], color=[C_ON, C_OFF],
                width=0.5, edgecolor='white', lw=1.5, zorder=3)
for bar, v in zip(bars3, [DROP_ON, DROP_OFF]):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.06,
             f'+{v:.3f}°', ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='#1A1A1A')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['KernelNet O', 'KernelNet X'], fontsize=11)
ax3.set_ylabel('Performance Drop (deg)\n(Frozen − Fine-tune)', fontsize=10)
ax3.set_title('Backbone Freezing Impact', fontsize=12, fontweight='bold', pad=8)
ax3.set_ylim(0, 4.8)
ax3.grid(axis='y', alpha=0.3, zorder=0)
ax3.spines[['top', 'right']].set_visible(False)

out = Path('results/plots/frozen_analysis.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f'saved: {out}')
