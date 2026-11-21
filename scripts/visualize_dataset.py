from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_pusht_dataset(repo_id: str = "lerobot/pusht", episodes: list[int] | None = None) -> LeRobotDataset:
    try:
        return LeRobotDataset(repo_id, episodes=episodes)
    except Exception as exc:
        print(f"[WARN] Direct HF dataset load failed ({exc}), retrying with hf-mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        return LeRobotDataset(repo_id, episodes=episodes)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    assets_dir, out_dir = root / "assets", root / "outputs"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_episodes = list(range(10))
    dataset = load_pusht_dataset("lerobot/pusht", episodes=target_episodes)

    # Read Parquet columns directly to avoid decoding video frames
    hf_ds = dataset.hf_dataset
    ep_indices = np.asarray(hf_ds["episode_index"])
    frame_indices = np.asarray(hf_ds["frame_index"])
    actions = np.asarray(hf_ds["action"], dtype=np.float32)
    states = np.asarray(hf_ds["observation.state"], dtype=np.float32)

    deltas = np.diff(actions, axis=0)
    same_ep = ep_indices[1:] == ep_indices[:-1]
    step_speeds = np.linalg.norm(deltas[same_ep], axis=1)

    stats = {
        "repo_id": "lerobot/pusht",
        "analyzed_episodes": [int(x) for x in np.unique(ep_indices)],
        "total_frames_analyzed": int(len(actions)),
        "action_dim": int(actions.shape[1]),
        "action_x": {
            "min": round(float(actions[:, 0].min()), 2),
            "max": round(float(actions[:, 0].max()), 2),
            "mean": round(float(actions[:, 0].mean()), 2),
            "std": round(float(actions[:, 0].std()), 2),
        },
        "action_y": {
            "min": round(float(actions[:, 1].min()), 2),
            "max": round(float(actions[:, 1].max()), 2),
            "mean": round(float(actions[:, 1].mean()), 2),
            "std": round(float(actions[:, 1].std()), 2),
        },
        "mean_step_displacement": round(float(step_speeds.mean()), 2),
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=160)
    fig.suptitle(
        "HuggingFace LeRobot (lerobot/pusht) — Action Trajectory & Distribution Analysis (10 Episodes)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    ax1 = axes[0, 0]
    for idx, ep in enumerate(target_episodes[:3]):
        mask = ep_indices == ep
        t = frame_indices[mask]
        ax1.plot(t, actions[mask, 0], color=palette[idx], linewidth=1.8, label=f"Ep {ep} Action X")
        ax1.plot(t, actions[mask, 1], color=palette[idx], linewidth=1.4, linestyle="--", alpha=0.8, label=f"Ep {ep} Action Y")
    ax1.set_title("Action Trajectory Over Time (Solid=X, Dashed=Y)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Timestep (Frame)")
    ax1.set_ylabel("Target Position (px, 0-512)")
    ax1.legend(ncol=2, fontsize=8, loc="best", frameon=True)

    ax2 = axes[0, 1]
    for idx, ep in enumerate(target_episodes[:5]):
        mask = ep_indices == ep
        ax2.plot(states[mask, 0], states[mask, 1], color=palette[idx], linewidth=1.6, alpha=0.85, label=f"Episode {ep}")
        ax2.scatter(states[mask, 0][0], states[mask, 1][0], color=palette[idx], marker="o", s=35, zorder=5)
        ax2.scatter(states[mask, 0][-1], states[mask, 1][-1], color=palette[idx], marker="X", s=50, zorder=5)
    ax2.set_title("2D End-Effector Workspace Paths (o=Start, X=End)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Agent Position X (px)")
    ax2.set_ylabel("Agent Position Y (px)")
    ax2.set_xlim(0, 512)
    ax2.set_ylim(512, 0)  # Match top-left image coordinate origin
    ax2.legend(fontsize=8, loc="best", frameon=True)

    ax3 = axes[1, 0]
    bins = np.linspace(0, 512, 32)
    ax3.hist(actions[:, 0], bins=bins, alpha=0.55, color="#2563eb", edgecolor="white", label=f"Action X (mean={stats['action_x']['mean']:.1f})")
    ax3.hist(actions[:, 1], bins=bins, alpha=0.55, color="#10b981", edgecolor="white", label=f"Action Y (mean={stats['action_y']['mean']:.1f})")
    ax3.axvline(stats["action_x"]["mean"], color="#1d4ed8", linestyle="--", linewidth=1.4)
    ax3.axvline(stats["action_y"]["mean"], color="#047857", linestyle="--", linewidth=1.4)
    ax3.set_title("Action Coordinate Value Distribution", fontsize=10.5, fontweight="bold")
    ax3.set_xlabel("Target Coordinate (px)")
    ax3.set_ylabel("Frame Count")
    ax3.legend(fontsize=8.5, frameon=True)

    ax4 = axes[1, 1]
    ep0_speed = np.linalg.norm(np.diff(actions[ep_indices == target_episodes[0]], axis=0), axis=1)
    ax4.plot(np.arange(len(ep0_speed)), ep0_speed, color="#7c3aed", linewidth=1.6, label="Ep 0 Step Displacement")
    ax4.axhline(
        float(np.mean(ep0_speed)),
        color="#ef4444",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean Step Speed ({np.mean(ep0_speed):.1f} px/step)",
    )
    ax4.fill_between(np.arange(len(ep0_speed)), ep0_speed, alpha=0.18, color="#7c3aed")
    ax4.set_title("Action Step Velocity Profile (Episode 0)", fontsize=10.5, fontweight="bold")
    ax4.set_xlabel("Timestep (Frame)")
    ax4.set_ylabel("L2 Action Delta (px / step)")
    ax4.legend(fontsize=8.5, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(assets_dir / "action_distribution.png"), dpi=160)
    plt.close(fig)
    print(f"[DONE] Saved action_distribution.png ({len(actions)} frames)")


if __name__ == "__main__":
    main()
