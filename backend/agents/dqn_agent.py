import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn_net import DeepQNetwork


class ReplayMemory:
    """
    Lưu tuple: (state, next_state, reward, done)
    """
    def __init__(self, capacity=30_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, next_state, reward, done):
        self.memory.append((state, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_ready(self, batch_size):
        """Chỉ train khi có đủ samples"""
        return len(self.memory) >= batch_size


class DQNAgent:
    """
    DQN Agent — grouped action, khớp với Colab notebook.
    Config là flat dict (colab_config.yaml):
      learning_rate, gamma, replay_capacity, batch_size,
      epsilon_start, epsilon_end, num_decay_epochs
    """
    def __init__(self, config, device='cpu'):
        self.device = device

        # Hyperparameters — flat config (Colab style)
        self.gamma        = config['gamma']
        self.batch_size   = config['batch_size']
        self.epsilon      = config['epsilon_start']
        self.epsilon_min  = config['epsilon_end']

        decay_epochs      = config['num_decay_epochs']
        self.epsilon_step = (self.epsilon - self.epsilon_min) / decay_epochs

        self.memory = ReplayMemory(capacity=config['replay_capacity'])

        # Networks: policy (w) và target (w⁻)
        self.policy_net = DeepQNetwork().to(self.device)
        self.target_net = DeepQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer & Loss
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning_rate']
        )
        self.loss_fn = nn.MSELoss()

    # ACT
    def act(self, next_states_dict):
        """
        Epsilon-greedy cho GROUPED action.
        next_states_dict: {(x, rot): [4 features]} từ get_possible_states()
        """
        actions  = list(next_states_dict.keys())
        features = list(next_states_dict.values())

        if not actions:
            return None, None

        # EXPLORATION
        if random.random() <= self.epsilon:
            idx = random.randrange(len(actions))
            return actions[idx], features[idx]

        # EXPLOITATION — feed tất cả features vào policy_net 1 lần
        feat_tensor = torch.FloatTensor(np.array(features)).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(feat_tensor)   # shape (N, 1)

        best_idx = torch.argmax(q_values).item()
        return actions[best_idx], features[best_idx]

    # REPLAY
    def replay(self):
        """
        1. Sample mini-batch từ replay buffer
        2. Tính Q-target: yi = r + γ·max Q̂(x', w⁻)·(1−done)
        3. Tính loss: L(w) = [yi − Q̂(x, w)]²
        4. Backprop + gradient clipping + optimizer step
        """
        if not self.memory.is_ready(self.batch_size):
            return 0.0

        batch = self.memory.sample(self.batch_size)
        states, next_states, rewards, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-target dùng target_net w⁻
        with torch.no_grad():
            next_q = self.target_net(next_states)

        target_q  = rewards + self.gamma * next_q * (1 - dones)
        current_q = self.policy_net(states)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (ổn định training)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """w⁻ ← w (hard update)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Giảm ε tuyến tính sau mỗi episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)