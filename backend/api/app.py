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

app = FastAPI(title="Tetris DRL API - Colab Experiment", version="2.0.0")

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

# Map reward_type -> model file
MODEL_MAP = {
    "game_score": ROOT_DIR / "weights" / "colab" / "dqn_tetris_grouped_game_score.pth",
    "heuristic":  ROOT_DIR / "weights" / "colab" / "dqn_tetris_grouped_heuristic.pth",
}

# State toan cuc
current_reward_type = config.get("reward_type", "game_score")
env       = TetrisEnv(render_mode="api", reward_type=current_reward_type)
device    = torch.device("cpu")
agent     = DQNAgent(config=config, device=device)
step_count = 0   # dem so buoc tung van

def load_model(reward_type: str):
    """Load model tuong ung voi reward_type vao agent hien tai."""
    global current_reward_type, step_count
    weights_path = MODEL_MAP.get(reward_type)
    if weights_path and weights_path.exists():
        agent.policy_net.load_state_dict(
            torch.load(str(weights_path), map_location=device)
        )
        agent.policy_net.eval()
        agent.epsilon = 0.0
        env.reward_type = reward_type
        current_reward_type = reward_type
        print(f"[OK] Switched to: {reward_type} ({weights_path.name})")
        return True
    print(f"[WARN] File not found: {weights_path}")
    return False


# Load model theo config khi khoi dong
load_model(current_reward_type)


# API Endpoints
@app.get("/")
def read_root():
    weights_path = MODEL_MAP.get(current_reward_type)
    return {
        "message":     "Tetris DRL API (Colab) dang hoat dong!",
        "model":       weights_path.name if weights_path else "random",
        "reward_type": current_reward_type,
    }


@app.post("/api/switch-model")
def switch_model(reward_type: str):
    """
    Doi model ngay khi dang chay — khong can restart server.
    reward_type: 'game_score' | 'heuristic'
    """
    if reward_type not in MODEL_MAP:
        return {"success": False, "error": f"reward_type phai la: {list(MODEL_MAP.keys())}"}

    success = load_model(reward_type)
    env.reset()  # Reset board khi doi model

    weights_path = MODEL_MAP.get(reward_type)
    return {
        "success":     success,
        "reward_type": current_reward_type,
        "model":       weights_path.name if weights_path else "not found",
    }


@app.get("/api/model-info")
def model_info():
    """Tra ve thong tin model hien tai."""
    weights_path = MODEL_MAP.get(current_reward_type)
    return {
        "reward_type":       current_reward_type,
        "model":             weights_path.name if weights_path else "random",
        "available_models":  list(MODEL_MAP.keys()),
    }


@app.get("/api/start")
def start_game():
    """Reset ban co va tra ve trang thai khoi dau"""
    global step_count
    step_count  = 0
    state, info = env.reset()
    board_matrix = env._get_state().tolist()
    safe_info    = {k: float(v) for k, v in info.items()}
    return {
        "status":      "started",
        "board":       board_matrix,
        "info":        safe_info,
        "steps":       step_count,
        "reward_type": current_reward_type,
    }


@app.post("/api/next-step")
def next_step():
    """Yeu cau AI di 1 buoc, cap nhat ban co va tra ve ket qua"""
    global step_count
    if env.engine.game_over:
        return {
            "status": "game_over",
            "steps":  step_count,
            "board":  env._get_state().tolist(),
            "info":   {k: float(v) for k, v in env._get_heuristic_stats().items()},
        }

    next_states_dict = env.get_possible_states()

    if not next_states_dict:
        env.engine.game_over = True
        return {
            "status": "game_over",
            "steps":  step_count,
            "board":  env._get_state().tolist(),
            "info":   {k: float(v) for k, v in env._get_heuristic_stats().items()},
        }

    action, _ = agent.act(next_states_dict)
    _, reward, done, _, info = env.step(action)
    step_count += 1

    action_str = f"Thả ở cột {action[0]}, Góc xoay {action[1]}"

    return {
        "status": "playing" if not done else "game_over",
        "action": action_str,
        "reward": float(reward),
        "steps":  step_count,
        "board":  env._get_state().tolist(),
        "info":   {k: float(v) for k, v in info.items()},
    }