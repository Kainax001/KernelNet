import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (no display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = Path("results")
OUT_DIR     = RESULTS_DIR / "plots"

PROPOSED_NAME = "proposed_e2e_v1"
BASELINE_NAME = "baseline_e2e_v1"

COLOR = {
    "proposed": {"train": "#2196F3", "val": "#0D47A1"},
    "baseline": {"train": "#FF9800", "val": "#E65100"},
}


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_metrics(run_name: str) -> dict:
    path = RESULTS_DIR / run_name / "metrics.tsv"
    train_err, val_err, train_loss, val_loss = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["phase"] in ("train", "e2e", "phase1", "phase2"):
                # train 행과 val 행이 epoch당 2줄씩 있음
                pass
    # epoch 별로 train/val 분리
    epochs_data = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ep   = int(row["epoch"])
            loss = float(row["loss"])
            err  = float(row["angular_err_deg"])
            if ep not in epochs_data:
                epochs_data[ep] = {}
            # 같은 epoch에서 첫 행=train, 두 번째 행=val (저장 순서 기준)
            if "train_loss" not in epochs_data[ep]:
                epochs_data[ep]["train_loss"] = loss
                epochs_data[ep]["train_err"]  = err
            else:
                epochs_data[ep]["val_loss"] = loss
                epochs_data[ep]["val_err"]  = err

    epochs = sorted(epochs_data.keys())
    return {
        "epochs":      epochs,
        "train_loss":  [epochs_data[e].get("train_loss", float("nan")) for e in epochs],
        "val_loss":    [epochs_data[e].get("val_loss",   float("nan")) for e in epochs],
        "train_err":   [epochs_data[e].get("train_err",  float("nan")) for e in epochs],
        "val_err":     [epochs_data[e].get("val_err",    float("nan")) for e in epochs],
    }


def load_per_subject(run_name: str) -> dict:
    path = RESULTS_DIR / run_name / "per_subject.tsv"
    subjects, means, stds = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["subject"] == "all":
                continue
            subjects.append(row["subject"])
            means.append(float(row["angular_err_mean_deg"]))
            stds.append(float(row["angular_err_std_deg"]))
    return {"subjects": subjects, "means": means, "stds": stds}


def load_summary() -> list:
    path = RESULTS_DIR / "summary.tsv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["run_name"] in (PROPOSED_NAME, BASELINE_NAME):
                rows.append(row)
    # 중복 제거 (run_name 기준 total_seconds 최솟값 유지 — 열 제한 이상치 제거)
    seen = {}
    for r in rows:
        name = r["run_name"]
        if name not in seen or float(r["total_seconds"]) < float(seen[name]["total_seconds"]):
            seen[name] = r
    return list(seen.values())


# ─── 플롯 함수 ────────────────────────────────────────────────────────────────

def plot_learning_curves(p_data: dict, b_data: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curves: Proposed vs Baseline", fontsize=14, fontweight="bold")

    for ax, metric, ylabel in [
        (axes[0], "err",  "Angular Error (deg)"),
        (axes[1], "loss", "Cosine Loss"),
    ]:
        ax.plot(p_data["epochs"], p_data[f"train_{metric}"],
                color=COLOR["proposed"]["train"], lw=1.5, alpha=0.7, label="Proposed train")
        ax.plot(p_data["epochs"], p_data[f"val_{metric}"],
                color=COLOR["proposed"]["val"],   lw=2,   label="Proposed val")
        ax.plot(b_data["epochs"], b_data[f"train_{metric}"],
                color=COLOR["baseline"]["train"], lw=1.5, alpha=0.7, label="Baseline train", linestyle="--")
        ax.plot(b_data["epochs"], b_data[f"val_{metric}"],
                color=COLOR["baseline"]["val"],   lw=2,   label="Baseline val",   linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(p_data["epochs"]))

    plt.tight_layout()
    path = out_dir / "learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_per_subject(p_sub: dict, b_sub: dict, out_dir: Path):
    subjects = p_sub["subjects"]
    x = np.arange(len(subjects))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_p = ax.bar(x - w/2, p_sub["means"], w, yerr=p_sub["stds"],
                    color=COLOR["proposed"]["val"], alpha=0.85,
                    capsize=4, label="Proposed", error_kw={"elinewidth": 1})
    bars_b = ax.bar(x + w/2, b_sub["means"], w, yerr=b_sub["stds"],
                    color=COLOR["baseline"]["val"], alpha=0.85,
                    capsize=4, label="Baseline", error_kw={"elinewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("Angular Error (deg)")
    ax.set_title("Per-Subject Test Angular Error (mean ± std)", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # 값 레이블
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
                color=COLOR["proposed"]["val"])
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7,
                color=COLOR["baseline"]["val"])

    plt.tight_layout()
    path = out_dir / "per_subject.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_summary(summary_rows: list, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Proposed vs Baseline — Final Comparison", fontsize=13, fontweight="bold")

    metrics = [
        ("test_angular_err_deg", "Test Angular Error (deg)", True),
        ("val_angular_err_deg",  "Val Angular Error (deg)",  True),
        ("total_seconds",        "Training Time (sec)",      False),
    ]

    for ax, (key, ylabel, lower_better) in zip(axes, metrics):
        names  = [r["run_name"].replace("_v1", "").replace("_e2e", "") for r in summary_rows]
        values = [float(r[key]) for r in summary_rows]
        colors = [COLOR["proposed"]["val"] if "proposed" in r["variant"] else COLOR["baseline"]["val"]
                  for r in summary_rows]

        bars = ax.bar(names, values, color=colors, alpha=0.85, width=0.5)
        ax.set_ylabel(ylabel)
        note = "(lower is better)" if lower_better else ""
        ax.set_title(f"{ylabel}\n{note}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.4f}" if val < 100 else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # 더 좋은 쪽 강조
        if lower_better:
            best_idx = int(np.argmin(values))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    path = out_dir / "summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_val_err_convergence(p_data: dict, b_data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(p_data["epochs"], p_data["val_err"],
            color=COLOR["proposed"]["val"], lw=2, label="Proposed val err")
    ax.plot(b_data["epochs"], b_data["val_err"],
            color=COLOR["baseline"]["val"], lw=2, linestyle="--", label="Baseline val err")

    # best epoch 표시
    p_best_ep = p_data["epochs"][int(np.argmin(p_data["val_err"]))]
    b_best_ep = b_data["epochs"][int(np.argmin(b_data["val_err"]))]
    ax.axvline(p_best_ep, color=COLOR["proposed"]["val"], lw=1, linestyle=":", alpha=0.6)
    ax.axvline(b_best_ep, color=COLOR["baseline"]["val"], lw=1, linestyle=":", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Angular Error (deg)")
    ax.set_title("Validation Error Convergence", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(p_data["epochs"]))

    plt.tight_layout()
    path = out_dir / "val_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", default=PROPOSED_NAME)
    parser.add_argument("--baseline", default=BASELINE_NAME)
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading metrics...")
    p_data = load_metrics(args.proposed)
    b_data = load_metrics(args.baseline)
    p_sub  = load_per_subject(args.proposed)
    b_sub  = load_per_subject(args.baseline)
    summary = load_summary()

    print(f"generating plots -> {out_dir}/")
    plot_learning_curves(p_data, b_data, out_dir)
    plot_val_err_convergence(p_data, b_data, out_dir)
    plot_per_subject(p_sub, b_sub, out_dir)
    plot_summary(summary, out_dir)

    print("done.")


if __name__ == "__main__":
    main()
