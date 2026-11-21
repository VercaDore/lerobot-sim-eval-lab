"""
Automated Multi-Episode Evaluation of Pretrained Diffusion Policy on PushT (HuggingFace LeRobot).

Outputs:
  - outputs/eval_report.json         : Structured evaluation metrics & per-episode logs
  - outputs/videos/*.mp4             : Individual rollout videos for best & worst episodes
  - assets/pusht_comparison.gif      : Side-by-side trajectory comparison GIF for README
  - assets/eval_summary_chart.png    : Per-episode reward & success-rate summary chart
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file


def _extract_legacy_stats(repo_id: str) -> dict[str, dict[str, torch.Tensor]]:
    """
    Reconstruct dataset_stats from legacy LeRobot v0.3 checkpoints where
    normalization buffers live inside `model.safetensors` instead of
    `policy_preprocessor.json` (introduced in LeRobot v0.4).
    """
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    sd = load_file(weights_path)
    stats: dict[str, dict[str, torch.Tensor]] = {}
    mapping = [
        ("observation.image", "normalize_inputs.buffer_observation_image"),
        ("observation.state", "normalize_inputs.buffer_observation_state"),
        ("action", "unnormalize_outputs.buffer_action"),
    ]
    for feat_key, prefix in mapping:
        feat_stats: dict[str, torch.Tensor] = {}
        for stat_name in ("mean", "std", "min", "max"):
            tensor_key = f"{prefix}.{stat_name}"
            if tensor_key in sd:
                feat_stats[stat_name] = sd[tensor_key]
        if feat_stats:
            stats[feat_key] = feat_stats
    return stats


def load_policy_with_fallback(repo_id: str, device: torch.device):
    """Load pretrained DiffusionPolicy and pre/post processors with hf-mirror & v0.3->v0.4 compatibility."""
    try:
        policy = DiffusionPolicy.from_pretrained(repo_id)
    except Exception as exc:
        print(f"[WARN] Direct HuggingFace load failed ({exc}). Switching to HF_ENDPOINT=https://hf-mirror.com ...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        policy = DiffusionPolicy.from_pretrained(repo_id)

    try:
        preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=repo_id)
    except FileNotFoundError:
        print(
            "[INFO] Legacy v0.3 checkpoint detected (no policy_preprocessor.json). "
            "Reconstructing v0.4 pre/post processors from model.safetensors buffers..."
        )
        legacy_stats = _extract_legacy_stats(repo_id)
        preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=legacy_stats)

    policy.to(device)
    policy.eval()
    return policy, preprocess, postprocess


def prepare_observation(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert gym_pusht observation dict into LeRobot batch format."""
    state = torch.from_numpy(obs["agent_pos"]).to(dtype=torch.float32)
    image = torch.from_numpy(obs["pixels"]).to(dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1)  # HWC -> CHW
    return {
        "observation.state": state.unsqueeze(0).to(device),
        "observation.image": image.unsqueeze(0).to(device),
    }


def annotate_frame(frame: np.ndarray, title: str, subtitle: str, border_color: tuple[int, int, int]) -> np.ndarray:
    """Add a clean header bar and colored border to a single RGB frame."""
    img = Image.fromarray(frame).resize((288, 288), Image.Resampling.BILINEAR)
    header_h = 54
    pad = 4
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + header_h + 2 * pad), border_color)
    draw = ImageDraw.Draw(canvas)
    # Dark header background
    draw.rectangle([pad, pad, img.width + pad, header_h + pad], fill=(24, 28, 36))
    draw.text((pad + 10, pad + 8), title, fill=(255, 255, 255))
    draw.text((pad + 10, pad + 30), subtitle, fill=(200, 214, 229))
    canvas.paste(img, (pad, header_h + pad))
    return np.asarray(canvas)


def build_comparison_gif(
    left_ep: dict[str, Any],
    right_ep: dict[str, Any],
    output_path: Path,
    fps: int = 15,
    stride: int = 2,
) -> None:
    """Create a side-by-side comparison GIF of two episodes (e.g. Best vs Challenging)."""
    left_frames = left_ep["frames"]
    right_frames = right_ep["frames"]
    left_rewards = left_ep["step_rewards"]
    right_rewards = right_ep["step_rewards"]

    max_len = max(len(left_frames), len(right_frames))
    combined_frames = []

    for i in range(0, max_len, stride):
        li = min(i, len(left_frames) - 1)
        ri = min(i, len(right_frames) - 1)

        l_status = "SUCCESS" if left_ep["success"] else "PARTIAL"
        r_status = "SUCCESS" if right_ep["success"] else "CHALLENGING / FAIL"

        l_color = (46, 204, 113) if left_ep["success"] else (243, 156, 18)
        r_color = (46, 204, 113) if right_ep["success"] else (231, 76, 60)

        l_img = annotate_frame(
            left_frames[li],
            f"Best: Ep {left_ep['episode']} (Seed {left_ep['seed']}) [{l_status}]",
            f"Step {li}/{left_ep['steps']} | Reward: {left_rewards[li]:.2f} (Max {left_ep['max_reward']:.2f})",
            l_color,
        )
        r_img = annotate_frame(
            right_frames[ri],
            f"Hardest: Ep {right_ep['episode']} (Seed {right_ep['seed']}) [{r_status}]",
            f"Step {ri}/{right_ep['steps']} | Reward: {right_rewards[ri]:.2f} (Max {right_ep['max_reward']:.2f})",
            r_color,
        )

        gap = np.full((l_img.shape[0], 10, 3), 255, dtype=np.uint8)
        combined_frames.append(np.concatenate([l_img, gap, r_img], axis=1))

    # Hold final frame briefly so viewers can read the outcome
    for _ in range(fps):
        combined_frames.append(combined_frames[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), combined_frames, fps=fps, loop=0)


def plot_eval_summary(episodes: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    """Generate a clean summary bar chart of per-episode max rewards and success rate."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)

    ep_ids = [e["episode"] for e in episodes]
    max_rewards = [e["max_reward"] for e in episodes]
    colors = ["#2ecc71" if e["success"] else "#e74c3c" for e in episodes]

    bars = ax.bar(ep_ids, max_rewards, color=colors, edgecolor="#2c3e50", linewidth=0.6, alpha=0.9)
    ax.axhline(0.95, color="#34495e", linestyle="--", linewidth=1.2, label="Success Threshold (~0.95)")

    for bar, val in zip(bars, max_rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2c3e50",
        )

    ax.set_title(
        f"LeRobot DiffusionPolicy on PushT ({summary['n_episodes']} Episodes) | "
        f"Success Rate: {summary['success_rate_pct']:.1f}% | "
        f"Avg Max Reward: {summary['avg_max_reward']:.3f} | "
        f"Avg Inference: {summary['avg_infer_ms']:.2f} ms/step",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Episode Index", fontsize=10)
    ax.set_ylabel("Max Coverage Reward (0.0 - 1.0)", fontsize=10)
    ax.set_xticks(ep_ids)
    ax.set_ylim(0, 1.15)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2ecc71", edgecolor="#2c3e50", label="Success"),
        Patch(facecolor="#e74c3c", edgecolor="#2c3e50", label="Fail / Incomplete"),
        plt.Line2D([0], [0], color="#34495e", linestyle="--", label="Success Threshold"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained DiffusionPolicy on PushT.")
    parser.add_argument("--repo-id", default="lerobot/diffusion_pusht", help="HuggingFace policy repo ID")
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed-start", type=int, default=1000, help="Starting random seed")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs"
    vid_dir = out_dir / "videos"
    assets_dir = root / "assets"
    vid_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Running on device: {device} ({gpu_name})")
    print(f"[INFO] Loading policy: {args.repo_id}")

    policy, preprocess, postprocess = load_policy_with_fallback(args.repo_id, device)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=args.max_steps, render_mode="rgb_array")

    episodes_meta: list[dict[str, Any]] = []
    episodes_full: list[dict[str, Any]] = []

    for ep_idx in range(1, args.n_episodes + 1):
        seed = args.seed_start + ep_idx - 1
        policy.reset()
        obs, info = env.reset(seed=seed)

        frames = [env.render()]
        step_rewards = [0.0]
        actions_log = []
        infer_times_ms = []

        terminated = False
        truncated = False
        step = 0
        t_ep_start = time.perf_counter()

        while not (terminated or truncated):
            batch = prepare_observation(obs, device)
            if preprocess is not None:
                batch = preprocess(batch)

            t0 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if postprocess is not None:
                action = postprocess(action)

            numpy_action = action.squeeze(0).detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(numpy_action)
            step += 1

            frames.append(env.render())
            step_rewards.append(float(reward))
            actions_log.append(numpy_action.tolist())

        elapsed = time.perf_counter() - t_ep_start
        max_reward = max(step_rewards)
        sum_reward = float(sum(step_rewards))
        is_success = bool(terminated or info.get("is_success", False) or max_reward >= 0.95)
        avg_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0

        ep_record = {
            "episode": ep_idx,
            "seed": seed,
            "success": is_success,
            "max_reward": round(max_reward, 4),
            "sum_reward": round(sum_reward, 4),
            "steps": step,
            "elapsed_sec": round(elapsed, 2),
            "avg_infer_ms": round(avg_infer_ms, 2),
        }
        episodes_meta.append(ep_record)
        episodes_full.append({**ep_record, "frames": frames, "step_rewards": step_rewards, "actions": actions_log})

        status_str = "SUCCESS" if is_success else "FAIL"
        print(
            f"[Ep {ep_idx:02d}/{args.n_episodes}] seed={seed} | {status_str:7s} | "
            f"max_reward={max_reward:.3f} | steps={step:3d} | infer={avg_infer_ms:.2f}ms/step"
        )

    env.close()

    successes = sum(1 for e in episodes_meta if e["success"])
    summary = {
        "policy_repo_id": args.repo_id,
        "env_id": "gym_pusht/PushT-v0",
        "device": str(device),
        "gpu_name": gpu_name,
        "n_episodes": args.n_episodes,
        "success_count": successes,
        "success_rate_pct": round(100.0 * successes / args.n_episodes, 2),
        "avg_max_reward": round(float(np.mean([e["max_reward"] for e in episodes_meta])), 4),
        "avg_steps": round(float(np.mean([e["steps"] for e in episodes_meta])), 2),
        "avg_infer_ms": round(float(np.mean([e["avg_infer_ms"] for e in episodes_meta])), 2),
    }

    report = {"summary": summary, "episodes": episodes_meta}
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Summary: {json.dumps(summary, ensure_ascii=False)}")
    print(f"[SAVED] Report -> {report_path}")

    # Pick best (highest reward, fewest steps) and hardest (lowest reward, most steps) episodes
    best_ep = max(episodes_full, key=lambda e: (e["success"], e["max_reward"], -e["steps"]))
    hardest_ep = min(episodes_full, key=lambda e: (e["success"], e["max_reward"], -e["steps"]))

    best_mp4 = vid_dir / f"ep_{best_ep['episode']:02d}_seed{best_ep['seed']}_best.mp4"
    hardest_mp4 = vid_dir / f"ep_{hardest_ep['episode']:02d}_seed{hardest_ep['seed']}_hardest.mp4"
    imageio.mimsave(str(best_mp4), best_ep["frames"], fps=20)
    imageio.mimsave(str(hardest_mp4), hardest_ep["frames"], fps=20)
    print(f"[SAVED] Videos -> {best_mp4.name}, {hardest_mp4.name}")

    gif_path = assets_dir / "pusht_comparison.gif"
    build_comparison_gif(best_ep, hardest_ep, gif_path, fps=15, stride=2)
    print(f"[SAVED] Comparison GIF -> {gif_path}")

    chart_path = assets_dir / "eval_summary_chart.png"
    plot_eval_summary(episodes_meta, summary, chart_path)
    print(f"[SAVED] Summary Chart -> {chart_path}")


if __name__ == "__main__":
    main()
