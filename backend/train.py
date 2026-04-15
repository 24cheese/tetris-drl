import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Import các module bạn đã viết
from tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

def create_directories():
    """Tạo thư mục để lưu tạ (weights) và biểu đồ (plots)"""
    os.makedirs("weights", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

def plot_learning_curve(rewards, losses, epsilons, filename):
    """Hàm vẽ biểu đồ quá trình học tập để báo cáo hội đồng"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Trục Y bên trái: Vẽ Reward (Màu xanh)
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(rewards, color=color, alpha=0.6, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    # Trục Y bên phải: Vẽ Epsilon (Màu cam)
    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Epsilon (Exploration Rate)', color=color)
    ax2.plot(epsilons, color=color, linestyle='dashed', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('DQN Learning Curve on Tetris')
    fig.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def main():
    create_directories()

    # 1. KHỞI TẠO THÀNH PHẦN
    # Không dùng render_mode để chạy ngầm siêu tốc trên Colab/CPU
    env = TetrisEnv(render_mode=None) 
    
    # Ép dùng GPU nếu có (rất quan trọng cho Colab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện. Đang sử dụng thiết bị: {device}")
    
    agent = DQNAgent(state_shape=(20, 10), num_actions=5, device=device)

    # 2. SIÊU THAM SỐ HUẤN LUYỆN
    EPISODES = 2000        # Số ván game tối đa AI sẽ chơi
    MAX_STEPS = 5000         # Giới hạn số bước mỗi ván (chống AI chơi kẹt vào vòng lặp vô hạn)
    TARGET_UPDATE_FREQ = 10  # Cập nhật Target Network sau mỗi 10 episodes

    # Biến theo dõi metrics
    history_rewards = []
    history_losses = []
    history_epsilons = []
    best_reward = -float('inf')

    # 3. VÒNG LẶP HUẤN LUYỆN CHÍNH (The RL Loop)
    start_time = datetime.now()

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            # Bước 1: Agent quan sát trạng thái và chọn hành động
            action = agent.act(state)

            # Bước 2: Đưa hành động vào môi trường, nhận lại kết quả
            next_state, reward, done, _, info = env.step(action)

            # Bước 3: Lưu trải nghiệm này vào bộ nhớ (Replay Buffer)
            agent.memory.push(state, action, reward, next_state, done)

            # Bước 4: Mạng Nơ-ron học (Backpropagation) từ bộ nhớ
            loss = agent.replay()

            # Bước 5: Chuyển sang trạng thái tiếp theo
            state = next_state
            
            total_reward += reward
            total_loss += loss
            steps += 1

        # KẾT THÚC 1 EPISODE (Ván game)
        # Giảm tỷ lệ ngẫu nhiên (Epsilon) để AI bắt đầu tin vào trí khôn của nó
        agent.decay_epsilon()

        # Lưu lại metrics
        history_rewards.append(total_reward)
        history_losses.append(total_loss / steps if steps > 0 else 0)
        history_epsilons.append(agent.epsilon)

        # Cập nhật Target Network định kỳ (Bí quyết giúp mạng không bị ngáo)
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Print log ra màn hình
        print(f"Episode: {episode}/{EPISODES} | Steps: {steps} | "
              f"Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
              f"Holes: {info['holes']} | Bumpiness: {info['bumpiness']:.1f}")

        # LƯU LẠI MODEL TỐT NHẤT
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), "weights/best_dqn_tetris.pth")
            print(f"  --> Đã lưu model tốt nhất (Reward: {best_reward:.1f})")

        # Cứ mỗi 100 episodes, tự động vẽ lại biểu đồ 1 lần
        if episode % 100 == 0:
            plot_learning_curve(history_rewards, history_losses, history_epsilons, f"learning_curve_ep{episode}")

    # Kết thúc toàn bộ quá trình
    time_taken = datetime.now() - start_time
    print(f"\nHuấn luyện hoàn tất sau: {time_taken}")
    plot_learning_curve(history_rewards, history_losses, history_epsilons, "learning_curve_final")
    env.close()

if __name__ == "__main__":
    main()