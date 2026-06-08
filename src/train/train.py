import argparse
import csv
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config.config import load_config
from src.data.dataset import build_dataloaders
from src.models.model import build_model


# ─── 손실 / 평가 ──────────────────────────────────────────────────────────────

def cosine_loss(g_hat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return (1.0 - (g_hat * g).sum(dim=1)).mean()


def angular_error(g_hat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    cos = (g_hat * g).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


# ─── 시드 ─────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Sequential 모드 phase 전환 ───────────────────────────────────────────────

def _freeze_except_regressor(model):
    for name, param in model.named_parameters():
        param.requires_grad = "regressor" in name


def _unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def _rebuild_optimizer(model, cfg: dict) -> AdamW:
    opt_cfg = cfg["optimizer"]
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["weight_decay"],
    )


# ─── 학습 / 평가 루프 ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, cfg, device, epoch, n_epochs):
    model.train()
    total_loss = total_err = n = 0

    pbar = tqdm(loader, desc=f"[{epoch:3d}/{n_epochs}] train",
                leave=False, dynamic_ncols=True)

    for L, R, H, G in pbar:
        L, R, H, G = L.to(device), R.to(device), H.to(device), G.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                G_hat = model(L, R, H)
                loss  = cosine_loss(G_hat, G)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            G_hat = model(L, R, H)
            loss  = cosine_loss(G_hat, G)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_norm"])
            optimizer.step()

        bs = L.size(0)
        total_loss += loss.item() * bs
        total_err  += angular_error(G_hat.detach(), G).sum().item()
        n += bs

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", err=f"{total_err/n:.2f}deg")

    return total_loss / n, total_err / n


@torch.no_grad()
def eval_epoch(model, loader, device, epoch, n_epochs, split="val"):
    model.eval()
    total_loss = total_err = n = 0

    pbar = tqdm(loader, desc=f"[{epoch:3d}/{n_epochs}] {split} ",
                leave=False, dynamic_ncols=True)

    for L, R, H, G in pbar:
        L, R, H, G = L.to(device), R.to(device), H.to(device), G.to(device)
        G_hat = model(L, R, H)
        loss  = cosine_loss(G_hat, G)
        bs    = L.size(0)
        total_loss += loss.item() * bs
        total_err  += angular_error(G_hat, G).sum().item()
        n += bs

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", err=f"{total_err/n:.2f}deg")

    return total_loss / n, total_err / n


# ─── TSV 유틸 ─────────────────────────────────────────────────────────────────

def _append_tsv(path: Path, row: dict, header: list):
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── 피험자별 평가 ────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_per_subject(model, hdf5_path: str, cfg: dict, device):
    model.eval()
    norm_cfg = cfg["dataset"]["normalization"]
    mean = np.array(norm_cfg["mean"], dtype=np.float32)[:, None, None]
    std  = np.array(norm_cfg["std"],  dtype=np.float32)[:, None, None]

    with h5py.File(hdf5_path, "r") as f:
        split_arr   = f["split"][:].astype(str)
        subject_arr = f["subject"][:].astype(str)
        test_idx    = np.where(split_arr == "test")[0]

        left_eyes  = f["left_eye"][test_idx]
        right_eyes = f["right_eye"][test_idx]
        gazes      = f["gaze"][test_idx]
        head_poses = f["head_pose"][test_idx]
        subjs      = subject_arr[test_idx]

    def normalize(img):
        return (img.astype(np.float32) / 255.0 - mean) / std

    results = {}
    bs = cfg["train"]["batch_size"]

    for start in range(0, len(test_idx), bs):
        end  = min(start + bs, len(test_idx))
        L_np = np.stack([normalize(left_eyes[i])  for i in range(start, end)])
        R_np = np.stack([normalize(right_eyes[i]) for i in range(start, end)])
        H_np = head_poses[start:end]
        G_np = gazes[start:end]

        L = torch.from_numpy(L_np).to(device)
        R = torch.from_numpy(R_np).to(device)
        H = torch.from_numpy(H_np).to(device)
        G = torch.from_numpy(G_np).to(device)

        errs = angular_error(model(L, R, H), G).cpu().numpy()
        for i, subj in enumerate(subjs[start:end]):
            results.setdefault(subj, []).append(errs[i])

    rows = []
    all_errs = []
    for subj in sorted(results):
        errs = results[subj]
        all_errs.extend(errs)
        rows.append({
            "subject":              subj,
            "n_samples":            len(errs),
            "angular_err_mean_deg": round(float(np.mean(errs)), 4),
            "angular_err_std_deg":  round(float(np.std(errs)),  4),
        })
    rows.append({
        "subject":              "all",
        "n_samples":            len(all_errs),
        "angular_err_mean_deg": round(float(np.mean(all_errs)), 4),
        "angular_err_std_deg":  round(float(np.std(all_errs)),  4),
    })
    return rows


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="experiments/runs/ 아래 yaml 이름")
    args = parser.parse_args()

    cfg = load_config(args.run)
    seed_everything(cfg["experiment"]["seed"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # 디렉터리 생성
    ckpt_dir   = Path(cfg["paths"]["checkpoint_dir"])
    result_dir = Path(cfg["paths"]["result_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # config 스냅샷 저장
    with open(result_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # wandb 초기화
    wb = None
    if cfg["wandb"]["enabled"]:
        import wandb
        wb = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"] or None,
            name=cfg["experiment"]["name"],
            tags=cfg["wandb"]["tags"],
            config=cfg,
        )

    # DataLoader / 모델
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델: {cfg['model']['variant']} | 파라미터: {total_params:,} | device: {device}")

    # Optimizer & Scheduler
    opt_cfg = cfg["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["weight_decay"],
    )

    n_epochs      = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"]["warmup_epochs"]

    if warmup_epochs > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=cfg["scheduler"]["eta_min"])
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=cfg["scheduler"]["eta_min"])

    # Sequential 모드 설정
    train_mode   = cfg["train"]["mode"]
    freeze_ratio = cfg["train"]["freeze_ratio"]
    phase1_end   = math.floor(freeze_ratio * n_epochs) if train_mode == "sequential" else 0
    in_phase1    = False

    if train_mode == "sequential":
        _freeze_except_regressor(model)
        optimizer = _rebuild_optimizer(model, cfg)
        in_phase1 = True
        print(f"Sequential 모드: Phase1 epoch 1~{phase1_end} (Regressor만 학습)")

    # 학습 루프
    metrics_path   = result_dir / "metrics.tsv"
    metrics_header = ["epoch", "phase", "loss", "angular_err_deg", "lr"]
    best_val_err   = float("inf")
    best_epoch     = 0
    t_start        = time.time()

    for epoch in range(1, n_epochs + 1):
        if train_mode == "sequential" and in_phase1 and epoch > phase1_end:
            _unfreeze_all(model)
            optimizer = _rebuild_optimizer(model, cfg)
            in_phase1 = False
            print(f"→ Phase2 시작 (epoch {epoch}): 전체 파라미터 fine-tune")

        phase  = "phase1" if in_phase1 else ("e2e" if train_mode == "e2e" else "phase2")
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_err = train_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch, n_epochs)
        val_loss,   val_err   = eval_epoch(model, val_loader, device, epoch, n_epochs)
        scheduler.step()

        print(f"[{epoch:3d}/{n_epochs}] {phase} | "
              f"train loss={train_loss:.4f} err={train_err:.2f}° | "
              f"val loss={val_loss:.4f} err={val_err:.2f}° | lr={cur_lr:.2e}")

        for split_name, loss, err in [("train", train_loss, train_err), ("val", val_loss, val_err)]:
            _append_tsv(metrics_path, {
                "epoch": epoch, "phase": phase,
                "loss": round(loss, 6), "angular_err_deg": round(err, 4), "lr": cur_lr,
            }, metrics_header)

        if wb is not None:
            wb.log({"epoch": epoch, "train/loss": train_loss, "train/angular_err": train_err,
                    "val/loss": val_loss, "val/angular_err": val_err, "lr": cur_lr})

        if val_err < best_val_err:
            best_val_err = val_err
            best_epoch   = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    total_secs = time.time() - t_start
    print(f"\n학습 완료. best epoch={best_epoch} val_err={best_val_err:.4f}°")

    # 테스트
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    test_loss, test_err = eval_epoch(model, test_loader, device, best_epoch, n_epochs, split="test")
    print(f"Test: loss={test_loss:.4f} err={test_err:.4f}°")

    # per_subject.tsv
    subj_rows   = eval_per_subject(model, cfg["paths"]["hdf5_output"], cfg, device)
    subj_header = ["subject", "n_samples", "angular_err_mean_deg", "angular_err_std_deg"]
    subj_path   = result_dir / "per_subject.tsv"
    with open(subj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=subj_header, delimiter="\t")
        writer.writeheader()
        writer.writerows(subj_rows)

    test_std = subj_rows[-1]["angular_err_std_deg"]

    # summary.tsv
    summary_header = [
        "run_name", "variant", "train_mode", "best_epoch",
        "val_loss", "val_angular_err_deg", "test_angular_err_deg", "test_std_deg",
        "total_params", "total_seconds",
    ]
    val_loss_at_best, _ = eval_epoch(model, val_loader, device, best_epoch, n_epochs, split="val")
    _append_tsv(Path(cfg["paths"]["summary_tsv"]), {
        "run_name":             cfg["experiment"]["name"],
        "variant":              cfg["model"]["variant"],
        "train_mode":           cfg["train"]["mode"],
        "best_epoch":           best_epoch,
        "val_loss":             round(val_loss_at_best, 6),
        "val_angular_err_deg":  round(best_val_err, 4),
        "test_angular_err_deg": round(test_err, 4),
        "test_std_deg":         test_std,
        "total_params":         total_params,
        "total_seconds":        round(total_secs, 1),
    }, summary_header)

    if wb is not None:
        import wandb
        artifact = wandb.Artifact(name=cfg["experiment"]["name"], type="results")
        artifact.add_file(str(subj_path))
        artifact.add_file(str(result_dir / "config_snapshot.yaml"))
        wb.log_artifact(artifact)
        wb.finish()

    print(f"결과 저장: {result_dir}")


if __name__ == "__main__":
    main()
