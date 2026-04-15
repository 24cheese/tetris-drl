import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Mạng Nơ-ron xấp xỉ hàm Q-value cho game Tetris.
    Input: Ma trận trạng thái bàn cờ (20x10)
    Output: Q-value cho 5 hành động (Trái, Phải, Xoay, Hard Drop, Rơi)
    """
    def __init__(self, input_shape=(20, 10), num_actions=5):
        super(DQN, self).__init__()
        
        # Trải phẳng ma trận 20x10 thành vector 200 chiều
        self.input_dim = input_shape[0] * input_shape[1]
        self.num_actions = num_actions
        
        # Kiến trúc Multi-Layer Perceptron (MLP)
        # Lớp ẩn 1: 200 nơ-ron -> 128 nơ-ron
        self.fc1 = nn.Linear(self.input_dim, 128)
        
        # Lớp ẩn 2: 128 nơ-ron -> 64 nơ-ron
        self.fc2 = nn.Linear(128, 64)
        
        # Lớp đầu ra: 64 nơ-ron -> 5 hành động
        self.out = nn.Linear(64, self.num_actions)

    def forward(self, x):
        """
        Quá trình lan truyền xuôi (Forward pass)
        x: Tensor chứa trạng thái bàn cờ
        """
        # Trải phẳng (Flatten) input: [batch_size, 20, 10] -> [batch_size, 200]
        x = x.view(-1, self.input_dim)
        
        # Đưa qua các lớp ẩn với hàm kích hoạt ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Lớp output (Không dùng hàm kích hoạt vì đây là giá trị Q-value số thực)
        q_values = self.out(x)
        
        return q_values