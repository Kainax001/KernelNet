import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── 파라미터 수 (단위: M) ─────────────────────────────────────────────────────
KERNELNET  = 197_771  / 1e6   # 0.198M
REG_PROP   = 264_195  / 1e6   # 0.264M  (hidden=256)
REG_BASE   = 1_055_747 / 1e6  # 1.056M  (hidden=1024)
BACKBONE   = 11_176_512 / 1e6 # 11.177M (ResNet18, shared)

C_KN   = '#1565C0'
C_PROP = '#2E7D32'
C_BASE = '#B71C1C'
C_BB   = '#546E7A'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Parameter Count Comparison: Proposed vs Baseline',
             fontsize=15, fontweight='bold', color='#1A237E', y=1.01)

# ── 왼쪽: Non-backbone 파라미터 세부 분해 (stacked bar) ───────────────────────
ax = axes[0]
ax.set_facecolor('#FAFAFA')
ax.set_title('Non-backbone Parameters (breakdown)', fontsize=12, fontweight='bold', pad=10)

x = np.array([0, 1])
labels = ['Proposed\n(KernelNet + Reg-256)', 'Baseline\n(Reg-1024)']

bar_kn  = ax.bar(x[0], KERNELNET, color=C_KN,   width=0.45, label='KernelNet',       zorder=3)
bar_rp  = ax.bar(x[0], REG_PROP,  color=C_PROP,  width=0.45, bottom=KERNELNET,
                 label='Regressor (hidden=256)', zorder=3)
bar_rb  = ax.bar(x[1], REG_BASE,  color=C_BASE,  width=0.45,
                 label='Regressor (hidden=1024)', zorder=3)

# 값 레이블
def label_bar(ax, bar, val, bottom=0, color='white'):
    cy = bottom + val / 2
    ax.text(bar.get_x() + bar.get_width() / 2, cy,
            f'{val*1000:.0f}K\n({val:.3f}M)',
            ha='center', va='center', fontsize=9, fontweight='bold', color=color)

label_bar(ax, bar_kn[0],  KERNELNET, 0,           color='white')
label_bar(ax, bar_rp[0],  REG_PROP,  KERNELNET,   color='white')
label_bar(ax, bar_rb[0],  REG_BASE,  0,           color='white')

# 합계 표시
total_prop = KERNELNET + REG_PROP
total_base = REG_BASE
ax.text(x[0], total_prop + 0.03, f'Total: {total_prop:.3f}M',
        ha='center', fontsize=10, fontweight='bold', color='#1B5E20')
ax.text(x[1], total_base + 0.03, f'Total: {total_base:.3f}M',
        ha='center', fontsize=10, fontweight='bold', color='#B71C1C')

# 배수 화살표
y_mid = (total_prop + total_base) / 2
ax.annotate('', xy=(x[1] - 0.25, total_base * 0.55),
            xytext=(x[0] + 0.25, total_prop * 0.55),
            arrowprops=dict(arrowstyle='<->', color='#FF6F00', lw=2))
ax.text(0.5, (total_prop + total_base) / 2 * 0.55,
        f'×{total_base/total_prop:.1f}',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='#FF6F00')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Parameters (M)', fontsize=11)
ax.set_ylim(0, total_base * 1.18)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

# ── 오른쪽: Backbone 포함 전체 파라미터 비교 ──────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#FAFAFA')
ax2.set_title('Total Parameters (including Backbone)', fontsize=12, fontweight='bold', pad=10)

x2 = np.array([0, 1])
labels2 = ['Proposed', 'Baseline']

# Backbone (공통)
b_bb_p = ax2.bar(x2[0], BACKBONE, color=C_BB,   width=0.45, label='Backbone (ResNet18)', zorder=3)
b_bb_b = ax2.bar(x2[1], BACKBONE, color=C_BB,   width=0.45, zorder=3)

# Non-backbone
b_np = ax2.bar(x2[0], total_prop, color=C_PROP, width=0.45, bottom=BACKBONE,
               label='Non-backbone (Proposed)', alpha=0.9, zorder=3)
b_nb = ax2.bar(x2[1], total_base, color=C_BASE, width=0.45, bottom=BACKBONE,
               label='Non-backbone (Baseline)', alpha=0.9, zorder=3)

# 합계
total_p_all = BACKBONE + total_prop
total_b_all = BACKBONE + total_base
for xi, total, color in [(x2[0], total_p_all, '#1B5E20'), (x2[1], total_b_all, '#B71C1C')]:
    ax2.text(xi, total + 0.1, f'{total:.2f}M',
             ha='center', fontsize=10, fontweight='bold', color=color)

# Backbone 영역 레이블
for xi in x2:
    ax2.text(xi, BACKBONE / 2, f'Backbone\n{BACKBONE:.2f}M',
             ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')

# Non-backbone 영역 레이블
ax2.text(x2[0], BACKBONE + total_prop / 2,
         f'{total_prop:.3f}M', ha='center', va='center',
         fontsize=8.5, color='white', fontweight='bold')
ax2.text(x2[1], BACKBONE + total_base / 2,
         f'{total_base:.3f}M', ha='center', va='center',
         fontsize=8.5, color='white', fontweight='bold')

ax2.set_xticks(x2)
ax2.set_xticklabels(labels2, fontsize=10)
ax2.set_ylabel('Parameters (M)', fontsize=11)
ax2.set_ylim(0, total_b_all * 1.12)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out = Path('results/plots/param_compare.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print(f'saved: {out}')
