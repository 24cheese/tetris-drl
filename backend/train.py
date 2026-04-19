from pathlib import Path
import torch
import yaml
from datetime import datetime

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path=None):
    """Đọc cấu hình từ file YAML"""
    if config_path is None:
        config_path = ROOT_DIR / "backend" / "experiments" / "configs" / "dqn_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def create_directories(config):
    """Tạo thư mục dựa trên đường dẫn cấu hình (luôn tạo ở Project Root)"""
    saved_path_rel = config['training']['saved_path']
    saved_path = ROOT_DIR / saved_path_rel
    saved_path.mkdir(parents=True, exist_ok=True)
    
    plots_path = ROOT_DIR / "backend" / "plots"
    plots_path.mkdir(exist_ok=True)
    return saved_path

def main():
    config = load_config()
    SAVED_PATH = create_directories(config)

    env = TetrisEnv(render_mode=None) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện. Đang sử dụng thiết bị: {device}")
    
    agent = DQNAgent(config=config, device=device)

    EPISODES = config['training']['num_epochs']
    MAX_STEPS = 5000
    TARGET_UPDATE_FREQ = 10 
    SAVE_INTERVAL = config['training']['save_interval']

    best_reward = -float('inf')
    start_time = datetime.now()

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            next_states_dict = env.get_possible_states()
            
            if not next_states_dict:
                break 
                
            action, next_state_features = agent.act(next_states_dict)
            _, reward, done, _, info = env.step(action)
            agent.memory.push(state, next_state_features, reward, done)
            loss = agent.replay()

            state = next_state_features
            total_reward += reward
            total_loss += loss
            steps += 1

        # Giảm tỷ lệ mò mẫm sau mỗi tập
        agent.decay_epsilon()

        # Cập nhật Target Network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        print(f"Episode: {episode}/{EPISODES} | Steps: {steps} | "
              f"Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
              f"Holes: {info['holes']} | Bumpiness: {info['bumpiness']}")

        # Lưu model nếu điểm cao kỷ lục
        if total_reward > best_reward:
            best_reward = total_reward
            best_model_path = os.path.join(SAVED_PATH, "best_dqn_tetris.pth")
            torch.save(agent.policy_net.state_dict(), best_model_path)
            print(f"  --> Đã lưu Best Model (Reward: {best_reward:.1f})")

        # Lưu model định kỳ (checkpoint)
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(SAVED_PATH, f"dqn_checkpoint_ep{episode}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            print(f"  --> Đã lưu Checkpoint tập {episode}")

    time_taken = datetime.now() - start_time
    print(f"\nHuấn luyện hoàn tất sau: {time_taken}")

if __name__ == "__main__":
    main()