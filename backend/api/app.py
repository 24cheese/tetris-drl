from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

import sys
import os
# Đảm bảo Python hiểu cấu trúc thư mục để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

app = FastAPI(title="Tetris DRL API", version="1.0.0")

# Cấu hình CORS để cho phép Frontend (React) gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên đổi thành domain cụ thể của frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo biến toàn cục cho Môi trường và Tác tử
env = TetrisEnv(render_mode='api')
device = torch.device("cpu") # Chạy suy luận (inference) trên API thì CPU là đủ nhanh
agent = DQNAgent(state_shape=(20, 10), num_actions=5, device=device)

# Load "bộ não" tốt nhất mà bạn đã train được
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "best_dqn_tetris.pth")
if os.path.exists(WEIGHTS_PATH):
    agent.policy_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    agent.policy_net.eval() # Bật chế độ suy luận, không train nữa
    # Tắt hoàn toàn tỷ lệ ngẫu nhiên để AI chơi bằng 100% thực lực
    agent.epsilon = 0.0 
    print(f"✅ Đã tải trọng số AI thành công từ: {WEIGHTS_PATH}")
else:
    print("⚠️ CẢNH BÁO: Không tìm thấy file trọng số. AI sẽ chơi ngẫu nhiên.")

@app.get("/")
def read_root():
    return {"message": "Tetris DRL API đang hoạt động!"}

@app.get("/api/start")
def start_game():
    """Reset bàn cờ và trả về trạng thái khởi đầu"""
    state, info = env.reset()
    
    # 1. Dùng _get_state() để lấy trạng thái có cả viên gạch đang lơ lửng, sau đó ép kiểu sang mảng Python
    board_matrix = env._get_state().tolist() 
    
    # 2. Ép kiểu NumPy (Holes, Bumpiness) sang kiểu số thực chuẩn của Python
    safe_info = {k: float(v) for k, v in info.items()}
    
    return {
        "status": "started",
        "board": board_matrix,
        "info": safe_info
    }

@app.post("/api/next-step")
def next_step():
    """Yêu cầu AI đi 1 bước, cập nhật bàn cờ và trả về kết quả"""
    # Xử lý an toàn nếu game đã kết thúc
    if env.engine.game_over:
        return {
            "status": "game_over", 
            "board": env._get_state().tolist(), 
            "info": {k: float(v) for k, v in env._get_heuristic_stats().items()}
        }

    # 1. Lấy trạng thái hiện tại
    current_state = env._get_state()
    
    # 2. AI suy nghĩ và đưa ra hành động
    action = agent.act(current_state)
    
    # 3. Môi trường thực thi hành động
    _, reward, done, _, info = env.step(action)
    
    # Map số ID hành động ra chữ
    action_names = {0: "Trái", 1: "Phải", 2: "Xoay", 3: "Rơi Nhanh", 4: "Chờ"}

    return {
        "status": "playing" if not done else "game_over",
        "action": action_names[action],
        "reward": float(reward), # Ép kiểu NumPy sang Float chuẩn
        "board": env._get_state().tolist(), # Ép mảng 2D NumPy sang List chuẩn
        "info": {k: float(v) for k, v in info.items()} # Ép kiểu NumPy sang Float chuẩn
    }