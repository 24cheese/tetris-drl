from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import yaml
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "backend"))

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

app = FastAPI(title="Tetris DRL API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
CONFIG_PATH = ROOT_DIR / "backend" / "experiments" / "configs" / "colab_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 4 Experiments từ Colab v6 
MODEL_MAP = {
    "game_score": {
        "path":        ROOT_DIR / "weights" / "colab" / "dqn_tetris_grouped_game_score.pth",
        "label":       "DQN · Game Score",
        "reward_type": "game_score",
        "color":       "#0075de",
        "description": "1 + lines² × 10 − 2·gameover",
    },
    "heuristic": {
        "path":        ROOT_DIR / "weights" / "colab" / "dqn_tetris_grouped_heuristic.pth",
        "label":       "DQN · Heuristic",
        "reward_type": "heuristic",
        "color":       "#7c3aed",
        "description": "Δf = −0.51H + 0.76L − 0.36O − 0.18B",
    },
    "ddqn_cur_game_score": {
        "path":        ROOT_DIR / "weights" / "colab" / "dqn_tetris_ddqn_cur_grouped_game_score.pth",
        "label":       "DDQN+CUR · Game Score",
        "reward_type": "game_score",
        "color":       "#ea5a0c",
        "description": "Double DQN + Curriculum · 1 + lines² × 10 − 2·gameover",
    },
    "ddqn_cur_heuristic": {
        "path":        ROOT_DIR / "weights" / "colab" / "dqn_tetris_ddqn_cur_grouped_heuristic.pth",
        "label":       "DDQN+CUR · Heuristic",
        "reward_type": "heuristic",
        "color":       "#1aae39",
        "description": "Double DQN + Curriculum · Δf heuristic",
    },
}

# State toàn cục
current_model_key  = "game_score"
env                = TetrisEnv(render_mode="api", reward_type="game_score")
device             = torch.device("cpu")
agent              = DQNAgent(config=config, device=device)
step_count         = 0
lines_total        = 0   # tổng lines xóa được trong 1 ván


def load_model(model_key: str) -> bool:
    """Load model tương ứng với model_key vào agent hiện tại."""
    global current_model_key
    meta = MODEL_MAP.get(model_key)
    if not meta:
        return False
    weights_path = meta["path"]
    if weights_path.exists():
        agent.policy_net.load_state_dict(
            torch.load(str(weights_path), map_location=device)
        )
        agent.policy_net.eval()
        agent.epsilon    = 0.0
        env.reward_type  = meta["reward_type"]
        current_model_key = model_key
        print(f"[OK] Switched to: {meta['label']} ({weights_path.name})")
        return True
    print(f"[WARN] File not found: {weights_path}")
    return False


# Load model theo config khi khởi động
load_model(current_model_key)


# API Endpoints

@app.get("/")
def read_root():
    meta = MODEL_MAP[current_model_key]
    return {
        "message":   "Tetris DRL API (Colab v6) đang hoạt động!",
        "model_key": current_model_key,
        "label":     meta["label"],
    }


@app.post("/api/switch-model")
def switch_model(model_key: str):
    """
    Đổi model ngay khi đang chạy — không cần restart server.
    model_key: 'game_score' | 'heuristic' | 'ddqn_cur_game_score' | 'ddqn_cur_heuristic'
    """
    if model_key not in MODEL_MAP:
        return {
            "success": False,
            "error":   f"model_key phải là một trong: {list(MODEL_MAP.keys())}",
        }

    success = load_model(model_key)
    env.reset()

    meta = MODEL_MAP[current_model_key]
    return {
        "success":   success,
        "model_key": current_model_key,
        "label":     meta["label"],
        "color":     meta["color"],
    }


@app.get("/api/model-info")
def model_info():
    """Trả về thông tin tất cả models và model đang dùng."""
    models = []
    for key, meta in MODEL_MAP.items():
        models.append({
            "key":         key,
            "label":       meta["label"],
            "color":       meta["color"],
            "description": meta["description"],
            "available":   meta["path"].exists(),
        })
    return {
        "current_model_key": current_model_key,
        "current_label":     MODEL_MAP[current_model_key]["label"],
        "models":            models,
    }


@app.get("/api/start")
def start_game():
    """Reset bàn cờ và trả về trạng thái khởi đầu."""
    global step_count, lines_total
    step_count   = 0
    lines_total  = 0
    state, info  = env.reset()
    board_matrix = env._get_state().tolist()
    safe_info    = {k: int(v) for k, v in info.items()}
    safe_info["lines_cleared"] = 0

    meta = MODEL_MAP[current_model_key]
    return {
        "status":    "started",
        "board":     board_matrix,
        "info":      safe_info,
        "steps":     step_count,
        "lines":     lines_total,
        "model_key": current_model_key,
        "label":     meta["label"],
        "color":     meta["color"],
    }


@app.post("/api/next-step")
def next_step():
    """Yêu cầu AI đi 1 bước, cập nhật bàn cờ và trả về kết quả."""
    global step_count, lines_total

    if env.engine.game_over:
        stats     = env._get_heuristic_stats()
        safe_info = {k: int(v) for k, v in stats.items()}
        safe_info["lines_cleared"] = 0
        return {
            "status": "game_over",
            "steps":  step_count,
            "lines":  lines_total,
            "board":  env._get_state().tolist(),
            "info":   safe_info,
        }

    next_states_dict = env.get_possible_states()

    if not next_states_dict:
        env.engine.game_over = True
        stats     = env._get_heuristic_stats()
        safe_info = {k: int(v) for k, v in stats.items()}
        safe_info["lines_cleared"] = 0
        return {
            "status": "game_over",
            "steps":  step_count,
            "lines":  lines_total,
            "board":  env._get_state().tolist(),
            "info":   safe_info,
        }

    action, _ = agent.act(next_states_dict)
    state, reward, done, _, info = env.step(action)
    step_count += 1

    # state[0] = lines_cleared từ bước vừa thực hiện
    lc = int(state[0])
    lines_total += lc

    action_str = f"Thả ở cột {action[0]}, Góc xoay {action[1]}"

    safe_info = {k: int(v) for k, v in info.items()}
    safe_info["lines_cleared"] = lc

    return {
        "status": "playing" if not done else "game_over",
        "action": action_str,
        "reward": float(reward),
        "steps":  step_count,
        "lines":  lines_total,
        "board":  env._get_state().tolist(),
        "info":   safe_info,
    }