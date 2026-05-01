from pathlib import Path
import torch
import os
import yaml
from datetime import datetime

from env.tetris_env import TetrisEnv
from agents.dqn_agent import DQNAgent

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    """Doc cau hinh tu file YAML (flat dict, Colab style)"""
    if config_path is None:
        config_path = ROOT_DIR / "backend" / "experiments" / "configs" / "colab_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Thu muc luu model
    saved_path = ROOT_DIR / config['saved_path']
    saved_path.mkdir(parents=True, exist_ok=True)

    reward_type = config.get('reward_type', 'game_score')
    env    = TetrisEnv(render_mode=None, reward_type=reward_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bat dau huan luyen. Thiet bi: {device} | Reward: {reward_type}")

    agent = DQNAgent(config=config, device=device)

    EPISODES          = config['num_episodes']
    MAX_STEPS         = 5000
    TARGET_UPDATE_FREQ = config['target_update_freq']
    SAVE_INTERVAL     = config['save_interval']

    best_reward = -float('inf')
    start_time  = datetime.now()

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        total_loss   = 0
        steps        = 0
        done         = False

        while not done and steps < MAX_STEPS:
            next_states_dict = env.get_possible_states()
            if not next_states_dict:
                break

            action, next_state_features = agent.act(next_states_dict)
            _, reward, done, _, info    = env.step(action)
            agent.memory.push(state, next_state_features, reward, done)
            loss = agent.replay()

            state         = next_state_features
            total_reward += reward
            total_loss   += loss
            steps        += 1

        agent.decay_epsilon()

        # Cap nhat Target Network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        print(f"Episode: {episode}/{EPISODES} | Steps: {steps} | "
              f"Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f} | "
              f"Holes: {info['holes']} | Bumpiness: {info['bumpiness']}")

        # Luu best model
        if total_reward > best_reward:
            best_reward = total_reward
            best_path   = saved_path / "best_dqn_tetris.pth"
            torch.save(agent.policy_net.state_dict(), str(best_path))
            print(f"  --> Luu Best Model (Reward: {best_reward:.1f})")

        # Luu checkpoint dinh ky
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            ckpt_path = saved_path / f"dqn_checkpoint_ep{episode}.pth"
            torch.save(agent.policy_net.state_dict(), str(ckpt_path))
            print(f"  --> Luu Checkpoint ep{episode}")

    print(f"\nHuan luyen hoan tat sau: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()