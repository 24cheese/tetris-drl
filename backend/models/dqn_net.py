import torch.nn as nn


class DeepQNetwork(nn.Module):
    """
    Xấp xỉ hàm q̂(x, u, w)
    Input : 4 features [lines_cleared, holes, bumpiness, total_height]
    Output: 1 Q-value scalar
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)