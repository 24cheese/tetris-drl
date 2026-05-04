"""
run_all_experiments.py
======================
Chạy đánh giá cả 4 model đã huấn luyện và xuất bảng thống kê so sánh.

4 experiments:
  1. DQN           + Game Score reward
  2. DQN           + Heuristic reward
  3. DDQN+Curriculum + Game Score reward
  4. DDQN+Curriculum + Heuristic reward

Kết quả được lưu vào:
  - results/experiment_results.json   (raw data)
  - results/experiment_summary.csv    (bảng CSV)
  - results/experiment_report.txt     (báo cáo đẹp in ra terminal)

Cách chạy:
  cd backend
  python run_all_experiments.py              
  python run_all_experiments.py --episodes 50  
"""

import argparse
import json
import csv
import os
import sys
import time
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from train import load_config

# Cấu hình 4 experiments
EXPERIMENTS = [
    {
        "id":          "dqn_game_score",
        "label":       "DQN – Game Score",
        "weight_file": "dqn_tetris_grouped_game_score.pth",
        "reward_type": "game_score",
    },
    {
        "id":          "dqn_heuristic",
        "label":       "DQN – Heuristic",
        "weight_file": "dqn_tetris_grouped_heuristic.pth",
        "reward_type": "heuristic",
    },
    {
        "id":          "ddqn_cur_game_score",
        "label":       "DDQN+Curriculum – Game Score",
        "weight_file": "dqn_tetris_ddqn_cur_grouped_game_score.pth",
        "reward_type": "game_score",
    },
    {
        "id":          "ddqn_cur_heuristic",
        "label":       "DDQN+Curriculum – Heuristic",
        "weight_file": "dqn_tetris_ddqn_cur_grouped_heuristic.pth",
        "reward_type": "heuristic",
    },
]

WEIGHTS_DIR = ROOT_DIR / "weights" / "colab"
RESULTS_DIR = ROOT_DIR / "backend" / "results"


# Helper

def stddev(data):
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return math.sqrt(sum((x - mean) ** 2 for x in data) / (len(data) - 1))


def confidence_interval_95(data):
    """95% CI bằng t-distribution xấp xỉ (z=1.96 với n>=30, t-table cho n nhỏ)."""
    n = len(data)
    if n < 2:
        return 0.0
    sd = stddev(data)
    # t-critical (two-tail 95%) cho các mức n phổ biến
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
               15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
    df = n - 1
    t = t_table.get(df) or (t_table.get(30) if df > 30 else t_table.get(max(k for k in t_table if k <= df), 2.0))
    return t * sd / math.sqrt(n)


# Core evaluation

def evaluate_single_experiment(exp: dict, config: dict, num_episodes: int, device) -> dict:
    """
    Chạy một model qua num_episodes ván, trả về dict thống kê đầy đủ.
    """
    weight_path = WEIGHTS_DIR / exp["weight_file"]
    if not weight_path.exists():
        print(f"  [SKIP] Không tìm thấy: {weight_path}")
        return None

    # Khởi tạo env với đúng reward_type
    env   = TetrisEnv(render_mode=None, reward_type=exp["reward_type"])
    agent = DQNAgent(config=config, device=device)

    state_dict = torch.load(str(weight_path), map_location=device)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.epsilon = 0.0   # greedy hoàn toàn

    ep_rewards     = []
    ep_lines       = []
    ep_steps       = []
    ep_holes       = []
    ep_bumpiness   = []

    print(f"\n  ▸ [{exp['label']}]  ({num_episodes} episodes)")
    print(f"    {'Ep':>4}  {'Pieces':>7}  {'Reward':>9}  {'Lines':>6}  {'Holes':>6}  {'Bump':>6}")
    print("    " + "─" * 54)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_line   = 0
        steps     = 0
        last_info = {}

        while not done:
            next_states_dict = env.get_possible_states()
            if not next_states_dict:
                break
            action, _ = agent.act(next_states_dict)
            _, reward, done, _, info = env.step(action)
            ep_reward += reward
            ep_line   += info.get("lines_cleared", 0)
            last_info  = info
            steps     += 1

        ep_rewards.append(ep_reward)
        ep_lines.append(int(ep_line))
        ep_steps.append(int(steps))
        ep_holes.append(int(last_info.get("holes", 0)))
        ep_bumpiness.append(int(last_info.get("bumpiness", 0)))

        print(f"    {ep:>4}  {steps:>7}  {ep_reward:>9.2f}  {ep_line:>6}  "
              f"{last_info.get('holes', 0):>6}  {last_info.get('bumpiness', 0):>6}")

    def stat(data):
        arr = np.array(data, dtype=float)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
            "ci95":   confidence_interval_95(data),
        }

    return {
        "id":          exp["id"],
        "label":       exp["label"],
        "reward_type": exp["reward_type"],
        "weight_file": exp["weight_file"],
        "num_episodes": num_episodes,
        "raw": {
            "rewards":   ep_rewards,
            "lines":     ep_lines,
            "steps":     ep_steps,
            "holes":     ep_holes,
            "bumpiness": ep_bumpiness,
        },
        "stats": {
            "reward":    stat(ep_rewards),
            "lines":     stat(ep_lines),
            "pieces":    stat(ep_steps),
            "holes":     stat(ep_holes),
            "bumpiness": stat(ep_bumpiness),
        },
    }


# Reporting

class NumpyEncoder(json.JSONEncoder):
    """Cho phép json.dump xử lý numpy int64/float64/ndarray."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _col(val, width=10):
    if isinstance(val, float):
        return f"{val:>{width}.2f}"
    return str(val).rjust(width)


def print_comparison_table(results: list[dict]):
    metrics = [
        ("reward",    "Reward (tổng)"),
        ("lines",     "Lines Cleared"),
        ("pieces",    "Pieces Placed"),
        ("holes",     "Holes cuối ván"),
        ("bumpiness", "Bumpiness cuối"),
    ]
    sub_cols = ["mean", "std", "min", "max", "median", "ci95"]

    header_label_w = 30

    print("\n" + "═" * 110)
    print(" BẢNG SO SÁNH THỐNG KÊ CÁC EXPERIMENT ".center(110, "═"))
    print("═" * 110)

    for metric_key, metric_name in metrics:
        print(f"\n  📊  {metric_name}")
        # Header
        row_h = f"  {'Model':<{header_label_w}}"
        for sc in sub_cols:
            row_h += f"  {sc:>10}"
        print("  " + "─" * (header_label_w + len(sub_cols) * 12 + 2))
        print(row_h)
        print("  " + "─" * (header_label_w + len(sub_cols) * 12 + 2))

        best_mean_val = None
        best_mean_idx = None
        for i, r in enumerate(results):
            m = r["stats"][metric_key]["mean"]
            if best_mean_val is None:
                best_mean_val = m
                best_mean_idx = i
            else:
                # Reward / lines / steps → higher better
                # holes / bumpiness      → lower better
                if metric_key in ("holes", "bumpiness"):
                    if m < best_mean_val:
                        best_mean_val = m
                        best_mean_idx = i
                else:
                    if m > best_mean_val:
                        best_mean_val = m
                        best_mean_idx = i

        for i, r in enumerate(results):
            tag = " ★" if i == best_mean_idx else "  "
            row = f"  {(r['label'] + tag):<{header_label_w}}"
            for sc in sub_cols:
                row += _col(r["stats"][metric_key][sc])
        print(row)

        for i, r in enumerate(results):
            tag = " ★" if i == best_mean_idx else "  "
            row = f"  {(r['label'] + tag):<{header_label_w}}"
            for sc in sub_cols:
                row += _col(r["stats"][metric_key][sc])
            print(row)
        print("  " + "─" * (header_label_w + len(sub_cols) * 12 + 2))

    print("\n  ★ = best mean trong metric đó")
    print("═" * 110 + "\n")


def save_json(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now().isoformat(), "experiments": results},
            f, ensure_ascii=False, indent=2, cls=NumpyEncoder
        )
    print(f"  ✔ JSON saved → {path}")


def save_csv(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["reward", "lines", "pieces", "holes", "bumpiness"]  # ← pieces thay steps
    sub     = ["mean", "std", "min", "max", "median", "ci95"]

    header = ["experiment", "reward_type"]
    for m in metrics:
        for s in sub:
            header.append(f"{m}_{s}")

    rows = []
    for r in results:
        row = [r["label"], r["reward_type"]]
        for m in metrics:
            for s in sub:
                row.append(round(r["stats"][m][s], 4))
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  ✔ CSV saved  → {path}")


def save_markdown(results: list[dict], path: Path):
    """Xuất báo cáo Markdown — dễ đọc, render được trên GitHub/Obsidian."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("reward",    "Reward (tổng)"),
        ("lines",     "Lines Cleared"),
        ("pieces",    "Pieces Placed"),
        ("holes",     "Holes cuối ván"),
        ("bumpiness", "Bumpiness cuối"),
    ]
    lines_out = []
    lines_out.append(f"# Tetris DRL — Experiment Results")
    lines_out.append(f"")
    lines_out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines_out.append(f"Episodes per model: {results[0]['num_episodes']}  ")
    lines_out.append(f"")

    for metric_key, metric_name in metrics:
        lines_out.append(f"## {metric_name}")
        lines_out.append(f"")
        # Header
        header = "| Model | Mean | Std | Min | Max | Median | CI 95% |"
        sep    = "|---|---|---|---|---|---|---|"
        lines_out.append(header)
        lines_out.append(sep)

        # Tìm best
        lower_better = metric_key in ("holes", "bumpiness")
        best_idx = min(range(len(results)), key=lambda i: results[i]["stats"][metric_key]["mean"]) \
                   if lower_better else \
                   max(range(len(results)), key=lambda i: results[i]["stats"][metric_key]["mean"])

        for i, r in enumerate(results):
            s   = r["stats"][metric_key]
            tag = " ★" if i == best_idx else ""
            lines_out.append(
                f"| **{r['label']}{tag}** "
                f"| {s['mean']:.2f} | {s['std']:.2f} "
                f"| {s['min']:.2f} | {s['max']:.2f} "
                f"| {s['median']:.2f} | ±{s['ci95']:.2f} |"
            )
        lines_out.append(f"")

    lines_out.append("> ★ = best mean trong metric đó")

    path.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"  ✔ Markdown   → {path}")


def save_txt_report(results: list[dict], path: Path):
    """Lưu bảng in ra terminal vào file .txt"""
    import io, contextlib
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison_table(results)
    path.write_text(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + buf.getvalue(),
        encoding="utf-8"
    )
    print(f"  ✔ TXT saved  → {path}")


# Entry point

def main():
    parser = argparse.ArgumentParser(description="Chạy và so sánh 4 Tetris DRL experiments")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Số episodes đánh giá mỗi model (default: 30)")
    parser.add_argument("--config",   type=str, default=None,
                        help="Đường dẫn file config YAML (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  TETRIS DRL — CHẠY & SO SÁNH 4 EXPERIMENTS")
    print("=" * 60)
    print(f"  Device   : {device}")
    print(f"  Episodes : {args.episodes} mỗi model")
    print(f"  Weights  : {WEIGHTS_DIR}")
    print("=" * 60)

    all_results = []
    total_start = time.time()

    for exp in EXPERIMENTS:
        t0 = time.time()
        result = evaluate_single_experiment(exp, config, args.episodes, device)
        elapsed = time.time() - t0
        if result is not None:
            result["elapsed_sec"] = round(elapsed, 1)
            all_results.append(result)
            avg_r = result["stats"]["reward"]["mean"]
            avg_l = result["stats"]["lines"]["mean"]
            print(f"\n  → Xong {exp['label']}: "
                  f"avg_reward={avg_r:.2f}, avg_lines={avg_l:.1f} "
                  f"({elapsed:.0f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n  Tổng thời gian: {total_elapsed:.0f}s")

    if not all_results:
        print("Không có kết quả nào. Kiểm tra đường dẫn weights.")
        return

    # In bảng so sánh
    print_comparison_table(all_results)

    # Lưu kết quả
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_csv(all_results,      RESULTS_DIR / f"{ts}_summary.csv")
    save_txt_report(all_results, RESULTS_DIR / f"{ts}_report.txt")
    save_markdown(all_results, RESULTS_DIR / f"{ts}_report.md")

    print("\n  Hoàn tất! Kết quả lưu tại:", RESULTS_DIR)


if __name__ == "__main__":
    main()
