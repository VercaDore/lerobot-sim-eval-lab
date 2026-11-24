from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from huggingface_hub import hf_hub_download
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from matplotlib.patches import Patch
from PIL import Image, ImageDraw
from safetensors.torch import load_file


def _extract_legacy_stats(repo_id: str) -> dict[str, dict[str, torch.Tensor]]:
    # LeRobot v0.3 checkpoints store normalization stats inside model.safetensors
    # rather than policy_preprocessor.json (required by v0.4).
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    sd = load_file(weights_path)
    stats: dict[str, dict[str, torch.Tensor]] = {}
    mapping = [
        ("observation.image", "normalize_inputs.buffer_observation_image"),
        ("observation.state", "normalize_inputs.buffer_observation_state"),
        ("action", "unnormalize_outputs.buffer_action"),
    ]
    for feat_key, prefix in mapping:
        feat_stats = {
            name: sd[f"{prefix}.{name}"]
            for name in ("mean", "std", "min", "max")
            if f"{prefix}.{name}" in sd
        }
        if feat_stats:
            stats[feat_key] = feat_stats
    return stats


def set_noise_scheduler(policy: DiffusionPolicy, scheduler: str, num_inference_steps: int | None) -> str:
    # The checkpoint was trained with a 100-step DDPM schedule. DDIM reuses the same betas, so it can sample
    # the same epsilon-prediction network with fewer steps (no retraining).
    diffusion = policy.diffusion
    if scheduler == "ddim":
        diffusion.noise_scheduler = DDIMScheduler.from_config(diffusion.noise_scheduler.config)
    n_steps = num_inference_steps or diffusion.noise_scheduler.config.num_train_timesteps
    diffusion.num_inference_steps = n_steps
    return f"{scheduler.upper()}-{n_steps}"


def load_policy_with_fallback(repo_id: str, device: torch.device):
    try:
        policy = DiffusionPolicy.from_pretrained(repo_id)
    except Exception as exc:
        print(f"[WARN] Direct HF load failed ({exc}), retrying with hf-mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        policy = DiffusionPolicy.from_pretrained(repo_id)

    try:
        preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=repo_id)
    except FileNotFoundError:
        print("[INFO] Legacy v0.3 checkpoint detected; rebuilding v0.4 processors from model.safetensors.")
        legacy_stats = _extract_legacy_stats(repo_id)
        preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=legacy_stats)

    policy.to(device)
    policy.eval()
    return policy, preprocess, postprocess


def prepare_observation(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.from_numpy(obs["agent_pos"]).to(dtype=torch.float32)
    image = torch.from_numpy(obs["pixels"]).to(dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1)  # HWC -> CHW
    return {
        "observation.state": state.unsqueeze(0).to(device),
        "observation.image": image.unsqueeze(0).to(device),
    }


def annotate_frame(frame: np.ndarray, title: str, subtitle: str, border_color: tuple[int, int, int]) -> np.ndarray:
    img = Image.fromarray(frame).resize((288, 288), Image.Resampling.BILINEAR)
    header_h, pad = 54, 4
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + header_h + 2 * pad), border_color)
    draw = ImageDraw.Draw(canvas)
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
    left_frames, right_frames = left_ep["frames"], right_ep["frames"]
    left_cov, right_cov = left_ep["step_coverage"], right_ep["step_coverage"]
    max_len = max(len(left_frames), len(right_frames))
    combined = []

    for i in range(0, max_len, stride):
        li = min(i, len(left_frames) - 1)
        ri = min(i, len(right_frames) - 1)

        l_status = "SUCCESS" if left_ep["success"] else "FAIL"
        r_status = "SUCCESS" if right_ep["success"] else "FAIL"
        l_color = (46, 204, 113) if left_ep["success"] else (231, 76, 60)
        r_color = (46, 204, 113) if right_ep["success"] else (231, 76, 60)

        l_img = annotate_frame(
            left_frames[li],
            f"Best: Ep {left_ep['episode']} (Seed {left_ep['seed']}) [{l_status}]",
            f"Step {li}/{left_ep['steps']} | Coverage: {left_cov[li]:.1%} (Max {left_ep['max_coverage']:.1%})",
            l_color,
        )
        r_img = annotate_frame(
            right_frames[ri],
            f"Hardest: Ep {right_ep['episode']} (Seed {right_ep['seed']}) [{r_status}]",
            f"Step {ri}/{right_ep['steps']} | Coverage: {right_cov[ri]:.1%} (Max {right_ep['max_coverage']:.1%})",
            r_color,
        )
        gap = np.full((l_img.shape[0], 10, 3), 255, dtype=np.uint8)
        combined.append(np.concatenate([l_img, gap, r_img], axis=1))

    combined.extend([combined[-1]] * fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), combined, fps=fps, loop=0)


def plot_eval_summary(episodes: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)

    ep_ids = [e["episode"] for e in episodes]
    max_cov = [e["max_coverage"] for e in episodes]
    colors = ["#2ecc71" if e["success"] else "#e74c3c" for e in episodes]

    bars = ax.bar(ep_ids, max_cov, color=colors, edgecolor="#2c3e50", linewidth=0.6, alpha=0.9)
    ax.axhline(SUCCESS_COVERAGE, color="#34495e", linestyle="--", linewidth=1.2)

    for bar, val in zip(bars, max_cov):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, color="#2c3e50")

    ax.set_title(
        f"DiffusionPolicy on PushT [{summary['sampler']}] ({summary['n_episodes']} eps) | "
        f"Success: {summary['success_rate_pct']:.1f}% | "
        f"Denoise pass: {summary['avg_denoise_ms']:.0f} ms | "
        f"Amortized: {summary['avg_step_ms']:.1f} ms/step",
        fontsize=10, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Episode Index", fontsize=10)
    ax.set_ylabel("Max Coverage of Goal Region", fontsize=10)
    ax.set_xticks(ep_ids)
    ax.set_ylim(0, 1.15)

    legend_handles = [
        Patch(facecolor="#2ecc71", edgecolor="#2c3e50", label="Success (env is_success)"),
        Patch(facecolor="#e74c3c", edgecolor="#2c3e50", label="Fail (timeout at max steps)"),
        plt.Line2D([0], [0], color="#34495e", linestyle="--", label="Success threshold: coverage > 0.95"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


SUCCESS_COVERAGE = 0.95  # gym_pusht: terminated = is_success = coverage > 0.95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/diffusion_pusht")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Denoising steps at inference (default: same as training, i.e. 100).")
    parser.add_argument("--tag", default="", help="Suffix for output files, e.g. '_ddim10'.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir, vid_dir, assets_dir = root / "outputs", root / "outputs" / "videos", root / "assets"
    vid_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    policy, preprocess, postprocess = load_policy_with_fallback(args.repo_id, device)
    sampler = set_noise_scheduler(policy, args.scheduler, args.num_inference_steps)
    print(f"[INFO] Device: {device} ({gpu_name}) | Policy: {args.repo_id} | Sampler: {sampler}")

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )

    episodes_meta: list[dict[str, Any]] = []
    episodes_full: list[dict[str, Any]] = []
    all_denoise_ms: list[float] = []
    all_step_ms: list[float] = []

    for ep_idx in range(1, args.n_episodes + 1):
        seed = args.seed_start + ep_idx - 1
        torch.manual_seed(seed)  # makes the diffusion sampling noise reproducible
        policy.reset()
        obs, info = env.reset(seed=seed)

        frames, step_coverage = [env.render()], [0.0]
        denoise_ms, step_ms = [], []
        is_success = False
        terminated = truncated = False
        step = 0
        t_ep_start = time.perf_counter()

        while not (terminated or truncated):
            batch = prepare_observation(obs, device)
            if preprocess is not None:
                batch = preprocess(batch)

            # select_action only runs a full denoising pass when its action queue is empty
            # (every n_action_steps calls); otherwise it pops a cached action.
            will_denoise = len(policy._queues["action"]) == 0
            t0 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            step_ms.append(dt_ms)
            if will_denoise:
                denoise_ms.append(dt_ms)

            if postprocess is not None:
                action = postprocess(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
            step += 1
            frames.append(env.render())
            step_coverage.append(float(info["coverage"]))
            # Success is decided ONLY by the environment (coverage > 0.95). See docs/errata.md (E1).
            is_success = is_success or bool(info["is_success"])

        elapsed = time.perf_counter() - t_ep_start
        max_coverage = max(step_coverage)
        # Skip the first denoising pass of episode 1 (CUDA warm-up).
        all_denoise_ms.extend(denoise_ms[1:] if ep_idx == 1 else denoise_ms)
        all_step_ms.extend(step_ms)

        ep_record = {
            "episode": ep_idx,
            "seed": seed,
            "success": is_success,
            "max_coverage": round(max_coverage, 4),
            "final_coverage": round(step_coverage[-1], 4),
            "steps": step,
            "elapsed_sec": round(elapsed, 2),
            "n_denoise_passes": len(denoise_ms),
            "avg_denoise_ms": round(float(np.mean(denoise_ms)), 2),
            "avg_step_ms": round(float(np.mean(step_ms)), 2),
        }
        episodes_meta.append(ep_record)
        episodes_full.append({**ep_record, "frames": frames, "step_coverage": step_coverage})
        print(
            f"[Ep {ep_idx:02d}/{args.n_episodes}] seed={seed} | {'SUCCESS' if is_success else 'FAIL':7s} | "
            f"max_cov={max_coverage:.3f} | steps={step:3d} | denoise={ep_record['avg_denoise_ms']:.0f}ms | "
            f"amortized={ep_record['avg_step_ms']:.1f}ms/step",
            flush=True,
        )

    env.close()

    successes = sum(1 for e in episodes_meta if e["success"])
    summary = {
        "policy_repo_id": args.repo_id,
        "env_id": "gym_pusht/PushT-v0",
        "sampler": sampler,
        "success_criterion": "env info['is_success'] (coverage > 0.95)",
        "device": str(device),
        "gpu_name": gpu_name,
        "n_episodes": args.n_episodes,
        "seeds": f"{args.seed_start}-{args.seed_start + args.n_episodes - 1}",
        "success_count": successes,
        "success_rate_pct": round(100.0 * successes / args.n_episodes, 2),
        "avg_max_coverage": round(float(np.mean([e["max_coverage"] for e in episodes_meta])), 4),
        "avg_steps": round(float(np.mean([e["steps"] for e in episodes_meta])), 2),
        "avg_denoise_ms": round(float(np.mean(all_denoise_ms)), 2),
        "avg_step_ms": round(float(np.mean(all_step_ms)), 2),
    }

    tag = args.tag
    (out_dir / f"eval_report{tag}.json").write_text(
        json.dumps({"summary": summary, "episodes": episodes_meta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_ep = max(episodes_full, key=lambda e: (e["success"], e["max_coverage"], -e["steps"]))
    hardest_ep = min(episodes_full, key=lambda e: (e["success"], e["max_coverage"], -e["steps"]))

    imageio.mimsave(str(vid_dir / f"ep_{best_ep['episode']:02d}_seed{best_ep['seed']}_best{tag}.mp4"), best_ep["frames"], fps=20)
    imageio.mimsave(str(vid_dir / f"ep_{hardest_ep['episode']:02d}_seed{hardest_ep['seed']}_hardest{tag}.mp4"), hardest_ep["frames"], fps=20)
    build_comparison_gif(best_ep, hardest_ep, assets_dir / f"pusht_comparison{tag}.gif", fps=15, stride=2)
    plot_eval_summary(episodes_meta, summary, assets_dir / f"eval_summary_chart{tag}.png")
    print(f"[DONE] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
