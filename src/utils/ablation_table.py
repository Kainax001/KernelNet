import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── 데이터 ────────────────────────────────────────────────────────────────────
#              KernelNet O        KernelNet X
# hidden=256   proposed_e2e_v1   ablation_b
# hidden=1024  ablation_a        baseline_e2e_v1

data = {
    ('256',  'O'): 1.4049,
    ('256',  'X'): 1.4326,
    ('1024', 'O'): 1.3490,
    ('1024', 'X'): 1.3587,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={'width_ratios': [1.1, 1]})
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('2×2 Ablation: KernelNet × hidden_dim',
             fontsize=15, fontweight='bold', color='#1A237E', y=1.02)

# ── 왼쪽: 2×2 표 ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 4); ax.set_ylim(0, 3.6)
ax.axis('off')
ax.set_facecolor('#F8F9FA')

C_KN_ON  = '#E3F2FD'   # KernelNet O
C_KN_OFF = '#FFEBEE'   # KernelNet X
C_HEAD   = '#37474F'
C_BEST   = '#1B5E20'
C_WORST  = '#B71C1C'

def cell(ax, x, y, w, h, text, sub='', bg='white', tc='black', bold=False, fs=11):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor='#90A4AE', linewidth=1.2)
    ax.add_patch(rect)
    ty = y + h/2 + (0.12 if sub else 0)
    ax.text(x + w/2, ty, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color=tc)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.18, sub, ha='center', va='center',
                fontsize=8, color='#546E7A')

cw, rh = 1.5, 0.9

# 헤더 행
cell(ax, 0,    2.7, 1.0,  0.8, '',              bg='#ECEFF1')
cell(ax, 1.0,  2.7, cw,   0.8, 'KernelNet  O',  bg='#1565C0', tc='white', bold=True)
cell(ax, 2.5,  2.7, cw,   0.8, 'KernelNet  X',  bg='#B71C1C', tc='white', bold=True)

# 헤더 열
cell(ax, 0, 2.7-rh,   1.0, rh, 'hidden\n256',  bg='#ECEFF1', tc='#37474F', bold=True, fs=10)
cell(ax, 0, 2.7-2*rh, 1.0, rh, 'hidden\n1024', bg='#ECEFF1', tc='#37474F', bold=True, fs=10)

# 데이터 셀
best = min(data.values())

configs = [
    ('256',  'O', 1.0,  2.7-rh),
    ('256',  'X', 2.5,  2.7-rh),
    ('1024', 'O', 1.0,  2.7-2*rh),
    ('1024', 'X', 2.5,  2.7-2*rh),
]
run_labels = {
    ('256',  'O'): 'proposed_e2e_v1',
    ('256',  'X'): 'ablation_b',
    ('1024', 'O'): 'ablation_a',
    ('1024', 'X'): 'baseline_e2e_v1',
}

for (hd, kn, cx, cy) in configs:
    val = data[(hd, kn)]
    bg  = C_KN_ON if kn == 'O' else C_KN_OFF
    tc  = C_BEST if val == best else ('black')
    star = ' ★' if val == best else ''
    cell(ax, cx, cy, cw, rh,
         f'{val:.4f}°{star}',
         sub=run_labels[(hd, kn)],
         bg=bg, tc=tc, bold=(val == best), fs=12)

# KernelNet 효과 브래킷 (오른쪽)
for row_y, hd in [(2.7 - rh, '256'), (2.7 - 2*rh, '1024')]:
    v_on  = data[(hd, 'O')]
    v_off = data[(hd, 'X')]
    delta = v_off - v_on
    mid_y = row_y + rh/2
    ax.annotate('', xy=(4.05, mid_y + 0.2), xytext=(4.05, mid_y - 0.2),
                arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=1.8))
    ax.text(4.25, mid_y, f'Δ{delta:.4f}°\n(KernelNet\neffect)',
            ha='left', va='center', fontsize=8, color='#FF6F00', fontweight='bold')

ax.set_xlim(0, 5.2)
ax.set_title('Angular Error (test, °)  — lower is better',
             fontsize=11, pad=8, color='#37474F')

# 범례
p1 = mpatches.Patch(color=C_KN_ON,  label='KernelNet O')
p2 = mpatches.Patch(color=C_KN_OFF, label='KernelNet X')
p3 = mpatches.Patch(color='#1B5E20', label='Best result ★')
ax.legend(handles=[p1, p2, p3], loc='lower left', fontsize=8.5,
          framealpha=0.9, bbox_to_anchor=(0, -0.02))

# ── 오른쪽: 그룹 막대 그래프 ──────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')

x = np.array([0, 1])
width = 0.32

vals_on  = [data[('256', 'O')], data[('1024', 'O')]]
vals_off = [data[('256', 'X')], data[('1024', 'X')]]

b1 = ax2.bar(x - width/2, vals_on,  width, color='#1565C0', label='KernelNet O', alpha=0.88, zorder=3)
b2 = ax2.bar(x + width/2, vals_off, width, color='#B71C1C', label='KernelNet X', alpha=0.88, zorder=3)

for bar, val in zip(list(b1) + list(b2), vals_on + vals_off):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f'{val:.4f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

# delta 화살표
for xi, hd in zip(x, ['256', '1024']):
    v_on  = data[(hd, 'O')]
    v_off = data[(hd, 'X')]
    ax2.annotate('', xy=(xi - width/2, v_on + 0.001),
                 xytext=(xi + width/2, v_off + 0.001),
                 arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=1.6))
    ax2.text(xi, max(v_on, v_off) + 0.018,
             f'Δ{v_off - v_on:.4f}°', ha='center', fontsize=9,
             color='#FF6F00', fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(['hidden = 256', 'hidden = 1024'], fontsize=11)
ax2.set_ylabel('Angular Error (°)', fontsize=11)
ax2.set_ylim(1.30, 1.50)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title('KernelNet Effect by hidden_dim', fontsize=11, pad=8)

plt.tight_layout()
out = Path('results/plots/ablation_table.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f'saved: {out}')
