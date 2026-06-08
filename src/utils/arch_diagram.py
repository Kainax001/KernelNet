import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor('#F8F9FA')

def box(ax, x, y, w, h, label, sub='', color='#1565C0', fontsize=10, textcolor='white'):
    bbox = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle='round,pad=0.01',
                          facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(bbox)
    ax.text(x, y+(0.025 if sub else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)
    if sub:
        ax.text(x, y-0.06, sub, ha='center', va='center',
                fontsize=7.5, color=textcolor, alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2, color='#455A64', lw=1.5, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'))

# ── ProposedModel ──────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal'); ax.axis('off')
ax.set_facecolor('#F8F9FA')
ax.set_title('ProposedModel  (Backbone 제외)', fontsize=13, fontweight='bold',
             pad=12, color='#1A237E')

# 입력
box(ax, 0.25, 0.93, 0.22, 0.06, 'L (Left Eye)',  'B x 3 x 64 x 64', color='#37474F', fontsize=9)
box(ax, 0.75, 0.93, 0.22, 0.06, 'R (Right Eye)', 'B x 3 x 64 x 64', color='#37474F', fontsize=9)

# KernelNet 영역
kn_bg = FancyBboxPatch((0.05, 0.56), 0.90, 0.29,
                        boxstyle='round,pad=0.01', facecolor='#E3F2FD',
                        edgecolor='#1565C0', linewidth=2, linestyle='--', zorder=2)
ax.add_patch(kn_bg)
ax.text(0.50, 0.875, 'KernelNet', ha='center', fontsize=12,
        fontweight='bold', color='#0D47A1', zorder=5)

# KernelNet 내부
box(ax, 0.25, 0.810, 0.28, 0.05, 'Shared Encoder',
    'Conv x3 + BN + LReLU + Pool -> (B,128)', color='#1565C0', fontsize=8)
box(ax, 0.75, 0.810, 0.28, 0.05, 'Shared Encoder',
    'Conv x3 + BN + LReLU + Pool -> (B,128)', color='#1565C0', fontsize=8)
arrow(ax, 0.25, 0.90, 0.25, 0.835)
arrow(ax, 0.75, 0.90, 0.75, 0.835)

box(ax, 0.50, 0.740, 0.22, 0.05, 'Feature Avg',
    'f = (f_L + f_R) / 2  ->  (B,128)', color='#1976D2', fontsize=8)
arrow(ax, 0.25, 0.785, 0.41, 0.740)
arrow(ax, 0.75, 0.785, 0.59, 0.740)

box(ax, 0.50, 0.673, 0.20, 0.045, 'FC1 + LReLU', '128 -> 512', color='#1976D2', fontsize=8)
arrow(ax, 0.50, 0.715, 0.50, 0.696)

box(ax, 0.50, 0.605, 0.28, 0.045, 'FC2  ->  reshape',
    '512 -> 75  ->  K: (B,3,5,5)', color='#1976D2', fontsize=8)
arrow(ax, 0.50, 0.651, 0.50, 0.628)

# DynamicFilter
box(ax, 0.25, 0.485, 0.30, 0.06, 'DynamicFilter',
    'depthwise conv2d  groups=B*C\n-> (B,3,64,64)', color='#6A1B9A', fontsize=8)
box(ax, 0.75, 0.485, 0.30, 0.06, 'DynamicFilter',
    'depthwise conv2d  groups=B*C\n-> (B,3,64,64)', color='#6A1B9A', fontsize=8)

# K -> DynamicFilter
arrow(ax, 0.38, 0.590, 0.25, 0.515, color='#6A1B9A', lw=1.8, rad=0.3)
arrow(ax, 0.62, 0.590, 0.75, 0.515, color='#6A1B9A', lw=1.8, rad=-0.3)
ax.text(0.10, 0.555, 'K', fontsize=9, color='#6A1B9A', fontweight='bold', style='italic')
ax.text(0.79, 0.555, 'K', fontsize=9, color='#6A1B9A', fontweight='bold', style='italic')

# L, R raw -> DynamicFilter (우회 경로)
ax.annotate('', xy=(0.11, 0.515), xytext=(0.11, 0.90),
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2,
                            connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(0.89, 0.515), xytext=(0.89, 0.90),
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.2,
                            connectionstyle='arc3,rad=0'))
ax.text(0.04, 0.70, 'L raw', fontsize=7, color='#90A4AE', rotation=90, va='center')
ax.text(0.92, 0.70, 'R raw', fontsize=7, color='#90A4AE', rotation=90, va='center')

# SiameseBackbone
box(ax, 0.25, 0.375, 0.28, 0.06, 'SiameseBackbone',
    'ResNet18 fine-tune (shared)\n-> f_L: (B,512)', color='#455A64', fontsize=8)
box(ax, 0.75, 0.375, 0.28, 0.06, 'SiameseBackbone',
    'ResNet18 fine-tune (shared)\n-> f_R: (B,512)', color='#455A64', fontsize=8)
arrow(ax, 0.25, 0.455, 0.25, 0.405)
arrow(ax, 0.75, 0.455, 0.75, 0.405)

# H
box(ax, 0.50, 0.375, 0.13, 0.055, 'H',
    'Head Pose\n(B,3)', color='#37474F', fontsize=9)

# Regressor
box(ax, 0.50, 0.240, 0.55, 0.085,
    'Regressor  (hidden_dim = 256)',
    '[f_L ; f_R ; H] -> (B,1027)\nFC1: 1027->256, LReLU, Dropout(0.3)\nFC2: 256->3,  L2-normalize',
    color='#1B5E20', fontsize=8.5)
arrow(ax, 0.25, 0.345, 0.31, 0.283)
arrow(ax, 0.75, 0.345, 0.69, 0.283)
arrow(ax, 0.50, 0.348, 0.50, 0.283)

# 출력
box(ax, 0.50, 0.130, 0.26, 0.055, 'G_hat',
    'unit vector  (B,3)', color='#1B5E20', fontsize=10)
arrow(ax, 0.50, 0.198, 0.50, 0.158)

# 파라미터 요약
ax.text(0.50, 0.038,
        'KernelNet: 197,771  |  Regressor: 264,195  |  Non-backbone: 461,966',
        ha='center', fontsize=8.5, color='#1B5E20',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#1B5E20', lw=1.2))

# ── BaselineModel ──────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal'); ax.axis('off')
ax.set_facecolor('#F8F9FA')
ax.set_title('BaselineModel  (Backbone 제외)', fontsize=13, fontweight='bold',
             pad=12, color='#B71C1C')

# 입력
box(ax, 0.25, 0.93, 0.22, 0.06, 'L (Left Eye)',  'B x 3 x 64 x 64', color='#37474F', fontsize=9)
box(ax, 0.75, 0.93, 0.22, 0.06, 'R (Right Eye)', 'B x 3 x 64 x 64', color='#37474F', fontsize=9)

# No Filter 표시
no_bg = FancyBboxPatch((0.08, 0.78), 0.84, 0.09,
                        boxstyle='round,pad=0.01', facecolor='#FFEBEE',
                        edgecolor='#C62828', linewidth=2, linestyle='--', zorder=2)
ax.add_patch(no_bg)
ax.text(0.50, 0.835, 'No Preprocessing Filter',
        ha='center', fontsize=12, fontweight='bold', color='#C62828', zorder=5)
ax.text(0.50, 0.800, 'Raw images passed directly to Backbone  (no dynamic kernel)',
        ha='center', fontsize=9, color='#C62828', zorder=5)
arrow(ax, 0.25, 0.90, 0.25, 0.780)
arrow(ax, 0.75, 0.90, 0.75, 0.780)

# SiameseBackbone
box(ax, 0.25, 0.680, 0.28, 0.06, 'SiameseBackbone',
    'ResNet18 fine-tune (shared)\n-> f_L: (B,512)', color='#455A64', fontsize=8)
box(ax, 0.75, 0.680, 0.28, 0.06, 'SiameseBackbone',
    'ResNet18 fine-tune (shared)\n-> f_R: (B,512)', color='#455A64', fontsize=8)
arrow(ax, 0.25, 0.780, 0.25, 0.710)
arrow(ax, 0.75, 0.780, 0.75, 0.710)

# H
box(ax, 0.50, 0.680, 0.13, 0.055, 'H',
    'Head Pose\n(B,3)', color='#37474F', fontsize=9)

# Regressor (1024 - 강조)
box(ax, 0.50, 0.515, 0.58, 0.105,
    'Regressor  (hidden_dim = 1024)',
    '[f_L ; f_R ; H] -> (B,1027)\nFC1: 1027->1024, LReLU, Dropout(0.3)\nFC2: 1024->3,  L2-normalize\n( Proposed Regressor 대비 파라미터 4x )',
    color='#B71C1C', fontsize=8.5)
arrow(ax, 0.25, 0.650, 0.31, 0.568)
arrow(ax, 0.75, 0.650, 0.69, 0.568)
arrow(ax, 0.50, 0.653, 0.50, 0.568)

# 출력
box(ax, 0.50, 0.390, 0.26, 0.055, 'G_hat',
    'unit vector  (B,3)', color='#B71C1C', fontsize=10)
arrow(ax, 0.50, 0.463, 0.50, 0.418)

# 파라미터 요약
ax.text(0.50, 0.300,
        'Regressor: 1,055,747  |  Non-backbone: 1,055,747',
        ha='center', fontsize=8.5, color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C', lw=1.2))

# 비교 바 차트 (인셋)
bax = fig.add_axes([0.565, 0.08, 0.37, 0.16])
labels_c = ['KernelNet\n(Proposed)', 'Regressor\n(Proposed)', 'Regressor\n(Baseline)']
vals_c   = [0.198, 0.264, 1.056]
colors_c = ['#1565C0', '#1B5E20', '#B71C1C']
bars_c = bax.bar(range(3), vals_c, color=colors_c, alpha=0.85, width=0.5)
for bar, v in zip(bars_c, vals_c):
    bax.text(bar.get_x()+bar.get_width()/2, v+0.01,
             f'{v:.3f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')
bax.set_xticks(range(3))
bax.set_xticklabels(labels_c, fontsize=8)
bax.set_ylabel('Params (M)', fontsize=8)
bax.set_title('Non-backbone Params', fontsize=9, fontweight='bold')
bax.grid(True, axis='y', alpha=0.3)
bax.set_ylim(0, 1.3)
bax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('results/plots/architecture_review.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print('saved: results/plots/architecture_review.png')
