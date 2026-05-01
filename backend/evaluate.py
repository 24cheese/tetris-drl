import os
import torch
import numpy as np
from pathlib import Path

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent
from train import load_config

ROOT_DIR = Path(__file__).resolve().parent.parent


def evaluate_model(model_path, config, num_episodes=10):
    """Bat mo hinh choi N van voi Epsilon = 0 va tra ve diem trung binh"""
    if not os.path.exists(model_path):
        print(f"Khong tim thay file: {model_path}")
        return None

    reward_type = config.get('reward_type', 'game_score')
    env    = TetrisEnv(render_mode=None, reward_type=reward_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(config=config, device=device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0.0   # AI choi bang thuc luc

    total_rewards = []
    total_lines   = []

    print(f"\nDang cham diem: {os.path.basename(model_path)}")
    print("-" * 50)

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done     = False
        ep_reward = 0
        ep_lines  = 0

        while not done:
            next_states_dict = env.get_possible_states()
            if not next_states_dict:
                break
            action, _ = agent.act(next_states_dict)
            _, reward, done, _, info = env.step(action)
            ep_reward += reward
            ep_lines  += info.get('lines_cleared', 0)

        total_rewards.append(ep_reward)
        total_lines.append(ep_lines)
        print(f"  Van {ep}/{num_episodes} | Diem: {ep_reward:.1f} | Dong da an: {ep_lines}")

    avg_reward = np.mean(total_rewards)
    print("-" * 50)
    print(f"KET QUA: Diem trung binh = {avg_reward:.1f}")
    return avg_reward


def main():
    config      = load_config()
    weights_dir = ROOT_DIR / config['saved_path']  # absolute path

    # Danh sach model can test (chinh sua theo cac checkpoint ban co)
    models_to_test = [
        "dqn_tetris_grouped_game_score.pth",
        "dqn_tetris_grouped_heuristic.pth",
    ]

    results = {}

    for model_name in models_to_test:
        model_path = str(weights_dir / model_name)
        avg_score  = evaluate_model(model_path, config, num_episodes=5)
        if avg_score is not None:
            results[model_name] = avg_score

    print("\n" + "=" * 50)
    print(" BANG XEP HANG ".center(50, "="))

    for rank, (name, score) in enumerate(
        sorted(results.items(), key=lambda x: x[1], reverse=True), 1
    ):
        print(f"  {rank}. {name} | Trung binh: {score:.1f}")


if __name__ == "__main__":
    main()