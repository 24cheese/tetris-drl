import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Import mạng Nơ-ron bạn vừa viết
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn_net import DQN

class ReplayMemory:
    """Bộ nhớ lưu trữ kinh nghiệm để AI học lại (Experience Replay)"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Lưu một bước đi vào bộ nhớ"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Lấy ngẫu nhiên một mẻ (batch) kinh nghiệm để huấn luyện"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Tác tử học tăng cường sử dụng Deep Q-Network"""
    def __init__(self, state_shape=(20, 10), num_actions=5, device='cpu'):
        self.device = device
        self.num_actions = num_actions
        
        # 1. SIÊU THAM SỐ (Hyperparameters - Cần giải thích khi vấn đáp)
        self.gamma = 0.99           # Hệ số chiết khấu (Tầm nhìn xa)
        self.epsilon = 1.0          # Tỷ lệ khám phá ban đầu (100% random)
        self.epsilon_min = 0.01     # Tỷ lệ khám phá tối thiểu
        self.epsilon_decay = 0.995  # Tốc độ giảm epsilon sau mỗi episode
        self.learning_rate = 1e-3   # Tốc độ học
        self.batch_size = 64        # Số lượng mẫu học mỗi lần
        
        self.memory = ReplayMemory(capacity=10000)

        # 2. KHỞI TẠO MẠNG NƠ-RON
        # Mạng chính: Dùng để đưa ra quyết định và cập nhật liên tục
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        
        # Mạng mục tiêu (Target Network): Giữ nguyên trọng số, thỉnh thoảng mới cập nhật
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net chỉ dùng để dự đoán, không train trực tiếp
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """Chọn hành động theo chiến thuật Epsilon-Greedy"""
        # Khám phá (Exploration): Chọn ngẫu nhiên
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
            
        # Khai thác (Exploitation): Dùng mạng Nơ-ron chọn hành động tốt nhất
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        """Huấn luyện mạng Nơ-ron (Cập nhật trọng số)"""
        # Nếu chưa đủ kinh nghiệm trong bộ nhớ thì chưa học
        if len(self.memory) < self.batch_size:
            return 0.0

        # Lấy một mẻ dữ liệu ngẫu nhiên
        batch = self.memory.sample(self.batch_size)
        
        # Tách dữ liệu ra thành các list riêng biệt
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển đổi sang Tensor của PyTorch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- PHẦN TOÁN HỌC CỐT LÕI CỦA DQN (Phương trình Bellman) ---
        
        # 1. Tính Q-value hiện tại của hành động đã chọn
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 2. Tính Q-value mục tiêu (Target Q) từ Target Network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Nếu game over (done=1), target Q chỉ là reward, không có tương lai
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # 3. Tính độ lệch (Loss) và cập nhật mạng
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Kẹp gradient (Gradient Clipping) để tránh bùng nổ trọng số
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy trọng số từ mạng chính sang mạng mục tiêu"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Giảm dần tỷ lệ ngẫu nhiên sau mỗi lần chơi"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay