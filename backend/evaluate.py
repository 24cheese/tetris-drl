import os
import torch
import numpy as np

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from train import load_config

def evaluate_model(model_path, config, num_episodes=10):
    """Bắt mô hình chơi N ván với Epsilon = 0 và trả về điểm trung bình"""
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file: {model_path}")
        return None

    env = TetrisEnv(render_mode=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(config=config, device=device)
    # Load model
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval() # Bật chế độ thi cử
    
    # ÉP EPSILON VỀ 0.0 (KHÔNG ĐI MÒ MẪM)
    agent.epsilon = 0.0

    total_rewards = []
    total_lines = []

    print(f"\n🎮 Đang chấm điểm mô hình: {os.path.basename(model_path)}")
    print("-" * 50)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_lines = 0

        while not done:
            next_states_dict = env.get_possible_states()
            if not next_states_dict:
                break
                
            action, _ = agent.act(next_states_dict)
            _, reward, done, _, info = env.step(action)
            
            ep_reward += reward
            # Tuỳ vào cách bạn lưu stats, có thể lấy số dòng đã ăn từ env
            ep_lines += info.get('lines_cleared', 0)

        total_rewards.append(ep_reward)
        total_lines.append(ep_lines)
        print(f"  Ván {ep}/{num_episodes} | Điểm: {ep_reward:.1f} | Dòng đã ăn: {ep_lines}")

    avg_reward = np.mean(total_rewards)
    print("-" * 50)
    print(f"🏆 KẾT QUẢ CHUNG CUỘC: Điểm trung bình = {avg_reward:.1f}")
    return avg_reward

def main():
    config = load_config()
    weights_dir = config['training']['saved_path']
    
    # Danh sách các model để đo hiệu suất
    models_to_test = [
        "dqn_checkpoint_ep500.pth",
        "dqn_checkpoint_ep1000.pth",
        "dqn_checkpoint_ep1500.pth",
        "dqn_checkpoint_ep2000.pth",
        "dqn_checkpoint_ep2500.pth",
        "dqn_checkpoint_ep3000.pth",
        "best_dqn_tetris.pth"
    ]

    results = {}

    for model_name in models_to_test:
        model_path = os.path.join(weights_dir, model_name)
        avg_score = evaluate_model(model_path, config, num_episodes=5) # Cho mỗi model chơi 5 ván
        if avg_score is not None:
            results[model_name] = avg_score

    print("\n" + "="*50)
    print(" BẢNG XẾP HẠNG (LEADERBOARD) ".center(50, "="))
    
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_results, 1):
        if rank == 1:
            print(f"🥇 Hạng 1: {name} (Trung bình: {score:.1f} điểm)")
        else:
            print(f" {rank}. {name} (Trung bình: {score:.1f} điểm)")

if __name__ == "__main__":
    main()