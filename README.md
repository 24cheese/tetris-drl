# 🧱 Tetris Deep Reinforcement Learning (DRL)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)

Một hệ thống ứng dụng **Deep Reinforcement Learning** để giải quyết trò chơi Tetris một cách tối ưu. Dự án bao gồm môi trường mô phỏng tùy chỉnh theo chuẩn Gymnasium, các thuật toán học máy tiên tiến (DQN, PPO).

---

## 🌟 Tính năng nổi bật

- **AI Agents Đa dạng**: Hỗ trợ thuật toán DQN (Deep Q-Network), PPO (Proximal Policy Optimization) và Heuristic Baseline.
- **Custom Environment**: Môi trường Tetris được xây dựng trên chuẩn `gymnasium` với hiệu năng cao.
- **Hệ thống Quản lý Weights**: Tách biệt file trọng số (Artifacts) giúp quản lý mô hình dễ dàng.
- **Phân tích Chuyên sâu**: Bộ Notebook đi kèm để khảo sát hàm Reward và đặc trưng (Features).

---

## 📂 Kiến trúc Dự án (Detailed)

Dự án được phân tách thành các module độc lập, hỗ trợ khả năng tái tạo thí nghiệm cao.

```text
TetrisDRL/
│
├── backend/                            # Toàn bộ logic Game, AI và API
│   ├── api/                            # Chứa mã nguồn server (FastAPI)
│   │   ├── app.py                      # File khởi chạy server chính
│   ├── env/                            # Môi trường Gymnasium (Decoupled)
│   │   ├── tetris_engine.py            # Logic lõi của trò chơi Tetris
│   │   └── tetris_env.py               # Class TetrisEnv chuẩn Gymnasium
│   ├── models/                         # Cấu trúc mạng Nơ-ron (PyTorch)
│   │   ├── dqn_net.py                  # Mạng MLP cho DQN
│   │   └── actor_critic_net.py         # Mạng Actor-Critic cho PPO
│   ├── agents/                         # Các thuật toán DRL
│   │   ├── dqn_agent.py                # DQN: Replay Buffer, Epsilon-Greedy
│   │   └── ppo_agent.py                # PPO: Policy Gradient, GAE
│   ├── utils/                          # Các module hỗ trợ dùng chung
│   │   └── state_extractor.py          # Tính 4-6 features (Holes, Bumpiness...)
│   ├── experiments/                    # Quản lý thí nghiệm (Reproducibility)
│   │   ├── configs/                    # File cấu hình hyperparameter (YAML)
│   │   │   └── dqn_config.yaml         # Hyperparams: lr, gamma, batch_size...
│   │   └── results/                    # Log và kết quả training raw
│   ├── train.py                        # Script huấn luyện chính
│   ├── evaluate.py                     # Script đánh giá model
│   └── requirements.txt                # Thư viện Python cần thiết
│
├── frontend/                           # Giao diện người dùng (React + Tailwind)
│   ├── src/
│   │   ├── components/                 # Các thành phần UI (Board, Stats Card...)
│   │   ├── App.jsx                     # Layout và logic chính
│   │   └── index.css                   
│   ├── package.json
│   └── vite.config.js
│
├── weights/                            # File model (.pth) - Project Artifacts
│   ├── dqn/                            # Trọng số cho thuật toán DQN
│   └── ppo/                            # Trọng số cho thuật toán PPO
│
├── notebooks/                          # Jupyter Notebooks phân tích
│   ├── 01_env_exploration.ipynb        # Kiểm tra môi trường & hành động
│   ├── 02_reward_analysis.ipynb        # Tinh chỉnh hàm Reward
│   └── 03_feature_analysis.ipynb       # Khảo sát đặc trưng trạng thái
│
├── tests/                              # Unit test cho hệ thống
│   └── test_env.py                     # Kiểm tra logic môi trường
│
├── docs/                               # Tài liệu học thuật & thiết kế
│   ├── architecture_design.md          # Thiết kế tổng thể
│   └── reward_function.md              # Lý luận thiết kế hàm Reward
│
├── scripts/                            # Script tự động hoá (chạy nhanh trên terminal)
│   ├── run_train_dqn.bat               # Chạy training DQN với config mặc định
│   ├── run_train_ppo.bat               # Chạy training PPO với config mặc định
│   ├── run_evaluate_all.bat            # Đánh giá tất cả agents, xuất CSV
│   └── run_compare.bat                 # So sánh kết quả các agents
│
├── .gitignore
└── README.md                           # Hướng dẫn này
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Backend (Python)
Đảm bảo bạn đã cài đặt Python 3.9+.

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r backend/requirements.txt
```

**Huấn luyện AI:**
```bash
python backend/train.py
```

**Chạy API Server:**
```bash
python backend/api/app.py
```

### 2. Frontend (React)
Yêu cầu Node.js 18+.

```bash
cd frontend
npm install
npm run dev
```

---

## 🧠 Đặc trưng & Hàm Thưởng (DRL Design)

Hệ thống sử dụng các đặc trưng cốt lõi để biểu diễn trạng thái:
- **Lines Cleared**: Số hàng vừa biến mất.
- **Holes**: Số lượng lỗ hổng bị kẹt dưới các khối gạch.
- **Bumpiness**: Độ gồ ghề của bề mặt gạch.
- **Height**: Tổng chiều cao của các cột gạch.

---

## 🎨 Giao diện Người dùng

Giao diện được xây dựng dựa trên triết lý thiết kế của **Notion**:
- **Minimalism**: Sử dụng tông màu trắng và xám ấm (`#f6f5f4`) làm chủ đạo.
- **Whisper Borders**: Đường viền `1px solid rgba(0,0,0,0.1)` tinh tế.
- **Real-time Monitoring**: Giám sát Reward, Holes và Action trực quan bằng Lucide Icons.

---