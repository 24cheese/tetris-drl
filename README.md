# 🧱 Tetris Deep Reinforcement Learning (DRL)

Một dự án ứng dụng Trí tuệ Nhân tạo (Deep Reinforcement Learning) để giải quyết trò chơi Tetris. Hệ thống được thiết kế theo kiến trúc MLOps chuẩn mực, bao gồm Custom Environment (Gymnasium), quản lý thử nghiệm thuật toán (DQN, PPO), và hệ thống giả lập trực quan qua Web Frontend.

---

## 🏗 Kiến trúc Hệ thống (Directory Tree)

Dự án được phân tách thành các module độc lập, quản lý cấu hình tập trung và hỗ trợ khả năng tái tạo thí nghiệm (reproducibility) cao.

```text
tetrisdrl/
│
├── backend/                            # Toàn bộ logic Game, AI và API nằm ở đây
│   │
│   ├── api/                            # Chứa mã nguồn server (FastAPI/Flask)
│   │   ├── app.py                      # File khởi chạy server API
│   │   └── routes.py                   # Các endpoint (ví dụ: /start, /next-step)
│   │
│   ├── env/                            # Môi trường Gymnasium (Decoupled)
│   │   ├── __init__.py
│   │   └── tetris_env.py               # Class TetrisEnv (step, reset, render)
│   │
│   ├── models/                         # Cấu trúc mạng Nơ-ron (PyTorch)
│   │   ├── dqn_net.py                  # Mạng MLP cho DQN
│   │   └── actor_critic_net.py         # Mạng Actor-Critic cho PPO
│   │
│   ├── agents/                         # Các thuật toán DRL
│   │   ├── base_agent.py               # Abstract base class (interface chung)
│   │   ├── heuristic_agent.py          # Rule-based (Pierre Dellacherie / Baseline)
│   │   ├── dqn_agent.py                # DQN: Replay Buffer, Epsilon-Greedy, Update
│   │   └── ppo_agent.py                # PPO: Policy Gradient, GAE, Clip objective
│   │
│   ├── utils/                          # Các module hỗ trợ dùng chung
│   │   ├── state_extractor.py          # Tính 6 features (Holes, Bumpiness, Height...)
│   │   ├── replay_buffer.py            # Experience Replay Buffer (cho DQN)
│   │   └── logger.py                   # Ghi metrics ra TensorBoard / CSV
│   │
│   ├── experiments/                    # Quản lý thí nghiệm (Reproducibility)
│   │   ├── configs/                    # File cấu hình hyperparameter (YAML)
│   │   │   ├── dqn_config.yaml         # Hyperparams: lr, gamma, batch_size, epsilon...
│   │   │   ├── ppo_config.yaml         # Hyperparams: lr, clip_eps, n_steps, epochs...
│   │   │   └── heuristic_config.yaml   # Trọng số heuristic features
│   │   │
│   │   └── results/                    # Kết quả raw từ mỗi lần chạy
│   │       ├── dqn_run_001/            # Log, CSV, checkpoint của run 001
│   │       ├── dqn_run_002/
│   │       └── ppo_run_001/
│   │
│   ├── train.py                        # Script huấn luyện (đọc config từ experiments/)
│   ├── evaluate.py                     # Script đánh giá, in ra điểm số trung bình
│   └── requirements.txt                # Thư viện Python (gymnasium, torch, fastapi...)
│
├── frontend/                           # Giao diện người dùng (React + Tailwind CSS)
│   ├── public/
│   └── src/
│       ├── components/                 # Các thành phần UI có thể tái sử dụng
│       │   ├── Board.jsx               # Component render lưới Tetris 20x10
│       │   ├── ControlPanel.jsx        # Cụm nút Play/Pause, chọn model (DQN/PPO)
│       │   └── AnalyticsView.jsx       # Hiển thị Q-value, số lỗ hổng real-time
│       │
│       ├── hooks/
│       │   └── useGameState.js         # Custom hook quản lý trạng thái từ API
│       │
│       ├── services/
│       │   └── apiClient.js            # Cấu hình Axios/Fetch để gọi Backend
│       │
│       ├── App.jsx                     # Layout chính của ứng dụng
│       └── index.css                   # Import Tailwind layers
│       │
│   ├── package.json
│   └── tailwind.config.js
│
├── notebooks/                          # Jupyter Notebooks phân tích & trực quan
│   ├── 01_env_exploration.ipynb        # Kiểm tra môi trường, thống kê reward
│   ├── 02_reward_analysis.ipynb        # Phân tích và tinh chỉnh hàm Reward
│   ├── 03_feature_analysis.ipynb       # Khảo sát 6 features (tương quan, phân phối)
│   └── 04_comparison.ipynb             # So sánh hiệu năng: Heuristic vs DQN vs PPO
│
├── tests/                              # Unit test & Smoke test
│   ├── test_env.py                     # Kiểm tra step(), reset(), render()
│   ├── test_state_extractor.py         # Kiểm tra tính đúng đắn của 6 features
│   └── test_agents.py                  # Smoke test: agents có chạy không bị crash
│
├── weights/                            # File model đã train (.pth), phân theo agent
│   ├── dqn/
│   │   ├── dqn_v1_best.pth
│   │   └── dqn_v2_best.pth
│   └── ppo/
│       └── ppo_v1_best.pth
│
├── docs/                               # Tài liệu học thuật của dự án
│   ├── architecture_design.md          # Thiết kế tổng thể hệ thống
│   ├── reward_function.md              # Công thức và lý luận thiết kế hàm Reward
│   ├── state_representation.md         # Mô tả chi tiết 6 features đặc trưng
│   └── results/                        # Biểu đồ & bảng kết quả cuối cùng (xuất hình)
│
├── scripts/                            # Script tự động hoá (chạy nhanh trên terminal)
│   ├── run_train_dqn.bat               # Chạy training DQN với config mặc định
│   ├── run_train_ppo.bat               # Chạy training PPO với config mặc định
│   ├── run_evaluate_all.bat            # Đánh giá tất cả agents, xuất CSV
│   └── run_compare.bat                 # So sánh kết quả các agents
│
├── .gitignore
└── README.md                           # Hướng dẫn setup, chạy backend/frontend/train

## 🚀 Cài đặt & Khởi chạy (Local)

### 1. Yêu cầu hệ thống
- Python 3.9+
- Node.js 18+ (Dành cho Frontend)
- Khuyến nghị sử dụng GPU (CUDA) để huấn luyện nhanh hơn.

### 2. Thiết lập Môi trường Backend (Python)
```bash
# Clone dự án
git clone [https://github.com/your-username/TetrisDRL.git](https://github.com/your-username/TetrisDRL.git)
cd TetrisDRL

# Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r backend/requirements.txt