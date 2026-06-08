import sys
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.config import load_config
from src.models.model import ProposedModel

OUT   = Path('results/plots')
OUT.mkdir(exist_ok=True)
CKPT  = Path('checkpoints/proposed_100ep/best.pt')
RUN   = 'proposed_100ep'
HDF5  = Path('data/processed/mpiigaze.h5')
N_SAMPLES = 200   # 분석에 사용할 샘플 수


def load_model(cfg):
    model = ProposedModel(cfg)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def normalize(img_chw, mean, std):
    img = img_chw.astype(np.float32) / 255.0
    return ((img - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


def gaussian_kernel_2d(size, sigma):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float32)


@torch.no_grad()
def extract_kernels(model, cfg):
    norm_cfg = cfg['dataset']['normalization']
    mean = np.array(norm_cfg['mean'], dtype=np.float32)
    std  = np.array(norm_cfg['std'],  dtype=np.float32)

    with h5py.File(HDF5, 'r') as f:
        split  = f['split'][:].astype(str)
        idx    = np.where(split == 'test')[0][:N_SAMPLES]
        lefts  = f['left_eye'][idx]
        rights = f['right_eye'][idx]
        subjs  = f['subject'][idx].astype(str)

    with h5py.File(HDF5, 'r') as f:
        head_poses = f['head_pose'][idx]   # (N,3) pitch/yaw/roll rad
        gazes      = f['gaze'][idx]        # (N,3) unit vector

    kernels, brightnesses, subjects = [], [], []

    for i in range(len(idx)):
        L_np = normalize(lefts[i],  mean, std)
        R_np = normalize(rights[i], mean, std)
        L = torch.from_numpy(L_np).unsqueeze(0)
        R = torch.from_numpy(R_np).unsqueeze(0)

        K = model.kernel_net(L, R)  # (1,3,5,5)
        kernels.append(K.squeeze(0).numpy())  # (3,5,5)

        gray = lefts[i].mean(0)
        brightnesses.append(gray.mean())
        subjects.append(subjs[i])

    return (np.stack(kernels),         # (N,3,5,5)
            np.array(brightnesses),    # (N,)
            np.array(subjects),        # (N,)
            head_poses.astype(np.float32),  # (N,3)
            gazes.astype(np.float32))       # (N,3)


def plot_kernel_overview(kernels, init_gauss, out_dir):
    """평균 커널, 분산, 초기 Gaussian 비교"""
    ch_names = ['B', 'G', 'R']
    mean_k = kernels.mean(0)   # (3,5,5)
    std_k  = kernels.std(0)    # (3,5,5)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('Learned Kernel Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3)

    vmax_mean = np.abs(mean_k).max()
    vmax_std  = std_k.max()

    for c in range(3):
        # 초기 Gaussian
        ax = fig.add_subplot(gs[c, 0])
        im = ax.imshow(init_gauss, cmap='RdBu_r', vmin=-vmax_mean, vmax=vmax_mean)
        ax.set_title(f'Init Gaussian\n(ch {ch_names[c]})', fontsize=8)
        ax.axis('off')

        # 평균 커널
        ax = fig.add_subplot(gs[c, 1])
        im = ax.imshow(mean_k[c], cmap='RdBu_r', vmin=-vmax_mean, vmax=vmax_mean)
        ax.set_title(f'Mean Kernel\n(ch {ch_names[c]})', fontsize=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 표준편차
        ax = fig.add_subplot(gs[c, 2])
        im2 = ax.imshow(std_k[c], cmap='Oranges', vmin=0, vmax=vmax_std)
        ax.set_title(f'Std Dev\n(ch {ch_names[c]})', fontsize=8)
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046)

        # 평균 커널 - Gaussian 차이
        diff = mean_k[c] - init_gauss
        ax = fig.add_subplot(gs[c, 3])
        im3 = ax.imshow(diff, cmap='RdBu_r',
                        vmin=-abs(diff).max(), vmax=abs(diff).max())
        ax.set_title(f'Mean - Init\n(ch {ch_names[c]})', fontsize=8)
        ax.axis('off')
        plt.colorbar(im3, ax=ax, fraction=0.046)

        # 커널 값 분포 히스토그램
        ax = fig.add_subplot(gs[c, 4])
        ax.hist(kernels[:, c, :, :].flatten(), bins=50,
                color='#1565C0', alpha=0.7, density=True)
        ax.axvline(init_gauss.flatten().mean(), color='red', lw=1.5,
                   linestyle='--', label='Gaussian center')
        ax.set_title(f'Value Dist\n(ch {ch_names[c]})', fontsize=8)
        ax.set_xlabel('Kernel value', fontsize=7)
        ax.legend(fontsize=6)

    plt.savefig(out_dir / 'kernel_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved: kernel_overview.png')


def plot_kernel_vs_brightness(kernels, brightnesses, out_dir):
    """밝기와 커널 특성의 상관관계"""
    # 커널의 중심 가중치 (2,2) vs 주변 가중치 평균
    center  = kernels[:, :, 2, 2].mean(1)          # (N,) 3채널 평균
    surround = np.concatenate([
        kernels[:, :, :2, :].reshape(len(kernels), -1),
        kernels[:, :, 3:, :].reshape(len(kernels), -1),
    ], axis=1).mean(1)                               # (N,)
    sharpness = center - surround                    # 양수=샤프닝, 음수=블러링

    # 밝기 기준 3분위 분류
    q33 = np.percentile(brightnesses, 33)
    q67 = np.percentile(brightnesses, 67)
    dark   = kernels[brightnesses <= q33]
    mid    = kernels[(brightnesses > q33) & (brightnesses <= q67)]
    bright = kernels[brightnesses > q67]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Kernel Characteristics vs Image Brightness', fontsize=13, fontweight='bold')

    # 밝기 vs sharpness 산점도
    axes[0].scatter(brightnesses, sharpness, alpha=0.4, s=20, color='#1565C0')
    z = np.polyfit(brightnesses, sharpness, 1)
    p = np.poly1d(z)
    x_line = np.linspace(brightnesses.min(), brightnesses.max(), 100)
    axes[0].plot(x_line, p(x_line), 'r--', lw=2, label=f'trend')
    corr = np.corrcoef(brightnesses, sharpness)[0, 1]
    axes[0].set_xlabel('Image Brightness (mean pixel)')
    axes[0].set_ylabel('Kernel Sharpness\n(center - surround weight)')
    axes[0].set_title(f'Brightness vs Sharpness (r={corr:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 밝기 3분위별 평균 커널 (R채널)
    for ax, group, label, color in [
        (axes[1], dark,   'Dark (bottom 33%)',   '#1A237E'),
        (axes[1], mid,    'Mid (33~67%)',         '#42A5F5'),
        (axes[1], bright, 'Bright (top 33%)',     '#FFCA28'),
    ]:
        pass

    titles = ['Dark (bottom 33%)', 'Mid (33~67%)', 'Bright (top 33%)']
    groups = [dark, mid, bright]
    colors_map = ['Blues', 'Greens', 'Oranges']
    vmax = max(np.abs(g.mean(0)[2]).max() for g in groups)

    for i, (grp, title, cmap) in enumerate(zip(groups, titles, colors_map)):
        mean_k = grp.mean(0)[2]  # R채널 평균 커널
        im = axes[1].imshow(mean_k, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            alpha=0) if False else None

    # 서브플롯으로 3개 커널 표시
    ax1_pos = axes[1].get_position()
    axes[1].set_visible(False)
    for gi, (grp, title) in enumerate(zip(groups, titles)):
        sub = fig.add_axes([ax1_pos.x0 + gi*(ax1_pos.width/3+0.01),
                            ax1_pos.y0, ax1_pos.width/3-0.005, ax1_pos.height])
        mk = grp.mean(0).mean(0)  # 3채널 평균 커널 (5,5)
        im = sub.imshow(mk, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        sub.set_title(title, fontsize=8)
        sub.axis('off')
        plt.colorbar(im, ax=sub, fraction=0.046, pad=0.04)

    # sharpness 분포 비교
    axes[2].hist(sharpness[brightnesses <= q33],   bins=30, alpha=0.6,
                 color='#1A237E', label='Dark',   density=True)
    axes[2].hist(sharpness[(brightnesses>q33)&(brightnesses<=q67)], bins=30,
                 alpha=0.6, color='#42A5F5', label='Mid', density=True)
    axes[2].hist(sharpness[brightnesses > q67],    bins=30, alpha=0.6,
                 color='#FFCA28', label='Bright', density=True)
    axes[2].axvline(0, color='black', lw=1.5, linestyle='--', label='neutral')
    axes[2].set_xlabel('Kernel Sharpness')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Sharpness Distribution by Brightness Group')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.savefig(out_dir / 'kernel_vs_brightness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved: kernel_vs_brightness.png')


def plot_kernel_samples(kernels, brightnesses, out_dir, n=8):
    """밝기 순으로 정렬한 대표 샘플 커널"""
    sorted_idx = np.argsort(brightnesses)
    sample_idx = np.linspace(0, len(sorted_idx)-1, n, dtype=int)
    selected   = sorted_idx[sample_idx]

    fig, axes = plt.subplots(3, n, figsize=(n*1.8, 6))
    fig.suptitle(f'Representative Kernels (sorted by brightness, n={n})',
                 fontsize=12, fontweight='bold')
    ch_names = ['B', 'G', 'R']
    vmax = np.abs(kernels[selected]).max()

    for j, s_idx in enumerate(selected):
        for c in range(3):
            ax = axes[c, j]
            im = ax.imshow(kernels[s_idx, c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            if j == 0:
                ax.set_ylabel(f'ch {ch_names[c]}', fontsize=8)
            if c == 0:
                ax.set_title(f'b={brightnesses[s_idx]:.1f}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_dir / 'kernel_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved: kernel_samples.png')


def kernel_sharpness(kernels):
    center   = kernels[:, :, 2, 2].mean(1)
    surround = np.concatenate([
        kernels[:, :, :2, :].reshape(len(kernels), -1),
        kernels[:, :, 3:, :].reshape(len(kernels), -1),
    ], axis=1).mean(1)
    return center - surround


def plot_kernel_vs_pose_gaze(kernels, head_poses, gazes, brightnesses, out_dir):
    sharpness = kernel_sharpness(kernels)

    # 분석 변수 정의
    pitch_abs = np.abs(np.degrees(head_poses[:, 0]))
    yaw_abs   = np.abs(np.degrees(head_poses[:, 1]))
    roll_abs  = np.abs(np.degrees(head_poses[:, 2]))
    head_mag  = np.sqrt(pitch_abs**2 + yaw_abs**2)   # pitch+yaw 크기

    # gaze angle from camera (arccos(-g_z))
    gz        = np.clip(gazes[:, 2], -1, 1)
    gaze_angle = np.degrees(np.arccos(-gz))
    gaze_x_abs = np.abs(np.degrees(np.arcsin(np.clip(gazes[:, 0], -1, 1))))
    gaze_y_abs = np.abs(np.degrees(np.arcsin(np.clip(gazes[:, 1], -1, 1))))

    variables = [
        (brightnesses, 'Image Brightness', 'blue'),
        (pitch_abs,    '|Head Pitch| (deg)', 'orange'),
        (yaw_abs,      '|Head Yaw| (deg)', 'green'),
        (head_mag,     'Head Pose Magnitude (deg)', 'purple'),
        (gaze_angle,   'Gaze Angle from Camera (deg)', 'red'),
        (gaze_x_abs,   '|Gaze Horizontal| (deg)', 'brown'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Kernel Sharpness vs Image/Pose/Gaze Variables',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for ax, (var, label, color) in zip(axes, variables):
        ax.scatter(var, sharpness, alpha=0.4, s=18, color=color)
        z = np.polyfit(var, sharpness, 1)
        x_line = np.linspace(var.min(), var.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', lw=1.8)
        r = np.corrcoef(var, sharpness)[0, 1]
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Kernel Sharpness', fontsize=9)
        ax.set_title(f'r = {r:.3f}', fontsize=11, fontweight='bold',
                     color='#B71C1C' if abs(r) > 0.3 else '#37474F')
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = out_dir / 'kernel_vs_pose_gaze.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved: {path}')

    # 상관계수 요약 출력
    print('\n── Correlation with Kernel Sharpness ──')
    for var, label, _ in variables:
        r = np.corrcoef(var, sharpness)[0, 1]
        print(f'  {label:<40s}: r = {r:+.3f}')


def main():
    cfg   = load_config(RUN)
    model = load_model(cfg)
    print(f'model loaded: {CKPT}')

    init_gauss = gaussian_kernel_2d(5, sigma=1.0)

    print(f'extracting kernels from {N_SAMPLES} test samples...')
    kernels, brightnesses, subjects, head_poses, gazes = extract_kernels(model, cfg)
    print(f'kernels shape: {kernels.shape}')
    print(f'brightness range: [{brightnesses.min():.2f}, {brightnesses.max():.2f}]')
    print(f'kernel value range: [{kernels.min():.4f}, {kernels.max():.4f}]')

    plot_kernel_overview(kernels, init_gauss, OUT)
    plot_kernel_vs_brightness(kernels, brightnesses, OUT)
    plot_kernel_samples(kernels, brightnesses, OUT)
    plot_kernel_vs_pose_gaze(kernels, head_poses, gazes, brightnesses, OUT)
    print('done.')


if __name__ == '__main__':
    main()
