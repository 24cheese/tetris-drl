from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import numpy as np
import yaml
import sys
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(ROOT_DIR / "backend"))

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

app = FastAPI(title="Tetris DRL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên đổi thành domain cụ thể của frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = ROOT_DIR / "backend" / "experiments" / "configs" / "dqn_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

env = TetrisEnv(render_mode='api')
device = torch.device("cpu")

agent = DQNAgent(config=config, device=device)

SAVED_PATH_REL = config['training']['saved_path']
WEIGHTS_PATH = ROOT_DIR / SAVED_PATH_REL / "best_dqn_tetris.pth"

if WEIGHTS_PATH.exists():
    agent.policy_net.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    agent.policy_net.eval() # Bật chế độ suy luận
    agent.epsilon = 0.0     # AI chơi 100% bằng thực lực
    print(f"✅ Đã tải trọng số AI thành công từ: {WEIGHTS_PATH}")
else:
    print(f"⚠️ CẢNH BÁO: Không tìm thấy file trọng số tại {WEIGHTS_PATH}. AI sẽ chơi ngẫu nhiên.")

@app.get("/")
def read_root():
    return {"message": "Tetris DRL API đang hoạt động!"}

@app.get("/api/start")
def start_game():
    """Reset bàn cờ và trả về trạng thái khởi đầu"""
    state, info = env.reset()
    
    # Lấy bàn cờ có cả gạch lơ lửng
    board_matrix = env._get_state().tolist() 
    
    safe_info = {k: float(v) for k, v in info.items()}
    
    return {
        "status": "started",
        "board": board_matrix,
        "info": safe_info
    }

@app.post("/api/next-step")
def next_step():
    """Yêu cầu AI đi 1 bước, cập nhật bàn cờ và trả về kết quả"""
    if env.engine.game_over:
        return {
            "status": "game_over", 
            "board": env._get_state().tolist(), 
            "info": {k: float(v) for k, v in env._get_heuristic_stats().items()}
        }

    next_states_dict = env.get_possible_states()
    
    if not next_states_dict: 
        env.engine.game_over = True
        return {
            "status": "game_over",
            "board": env._get_state().tolist(),
            "info": {k: float(v) for k, v in env._get_heuristic_stats().items()}
        }

    action, _ = agent.act(next_states_dict)
    _, reward, done, _, info = env.step(action)
    
    action_str = f"Thả ở Cột {action[0]}, Góc xoay {action[1]}"

    return {
        "status": "playing" if not done else "game_over",
        "action": action_str,
        "reward": float(reward), 
        "board": env._get_state().tolist(), 
        "info": {k: float(v) for k, v in info.items()} 
    }