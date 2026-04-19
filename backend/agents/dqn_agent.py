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
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, next_state, reward, done):
        self.memory.append((state, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, config, device='cpu'):
        self.device = device
        
        # Đọc tham số từ file config YAML
        self.gamma = config['agent']['gamma']
        self.learning_rate = config['agent']['learning_rate']
        self.batch_size = config['agent']['batch_size']
        
        self.epsilon = config['exploration']['initial_epsilon']
        self.epsilon_min = config['exploration']['final_epsilon']
        
        # Tính toán bước giảm Epsilon tuyến tính (Linear Decay)
        # Ví dụ: từ 1.0 xuống 0.001 trong 2000 vòng -> mỗi vòng giảm ~0.0005
        decay_epochs = config['exploration']['num_decay_epochs']
        self.epsilon_step = (self.epsilon - self.epsilon_min) / decay_epochs
        
        self.memory = ReplayMemory(capacity=config['agent']['replay_memory_size'])

        self.policy_net = DeepQNetwork().to(self.device)
        self.target_net = DeepQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, next_states_dict):
        actions = list(next_states_dict.keys())
        features = list(next_states_dict.values())

        if not actions: 
            return None, None

        # Epsilon-Greedy
        if random.random() <= self.epsilon:
            idx = random.randrange(len(actions))
            return actions[idx], features[idx]

        features_tensor = torch.FloatTensor(np.array(features)).to(self.device)
        
        with torch.no_grad():
            predictions = self.policy_net(features_tensor)
        
        best_index = torch.argmax(predictions).item()
        return actions[best_index], features[best_index]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        states, next_states, rewards, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states)
        
        with torch.no_grad():
            next_q = self.target_net(next_states)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Giảm epsilon tuyến tính theo từng tập (episode)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step
            self.epsilon = max(self.epsilon, self.epsilon_min)