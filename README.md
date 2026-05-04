# 🧱 Tetris Deep Reinforcement Learning (DRL)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)

Hệ thống ứng dụng **Deep Reinforcement Learning** huấn luyện AI chơi Tetris, so sánh hiệu quả của 4 cấu hình thí nghiệm khác nhau. Dự án bao gồm môi trường mô phỏng tùy chỉnh theo chuẩn Gymnasium, backend FastAPI, và dashboard React để theo dõi trực tiếp.

---

## 🌟 Tính năng

- **4 Experiments**: So sánh DQN và DDQN+Curriculum với 2 hàm reward (Game Score vs Heuristic)
- **Custom Tetris Env**: Môi trường Tetris chuẩn `gymnasium` với grouped action space
- **Dashboard Real-time**: Giao diện React theo dõi AI chơi, hiển thị stats từng bước
- **Evaluation Script**: Chạy & so sánh 4 model tự động, xuất báo cáo CSV / TXT / Markdown

---

## 📂 Cấu trúc Dự án

```text
tetris-drl/
│
├── backend/
│   ├── api/
│   │   └── app.py                  # FastAPI server — WebSocket + REST endpoints
│   ├── env/
│   │   ├── tetris_engine.py        # Logic lõi Tetris (board, pieces, collision)
│   │   └── tetris_env.py           # TetrisEnv chuẩn Gymnasium
│   ├── models/
│   │   └── dqn_net.py              # Mạng MLP (DeepQNetwork)
│   ├── agents/
│   │   └── dqn_agent.py            # DQNAgent: Replay Buffer, ε-greedy, target net
│   ├── experiments/
│   │   └── configs/
│   │       └── colab_config.yaml   # Hyperparameters (lr, gamma, batch_size...)
│   ├── results/                    # Kết quả evaluate (tự sinh)
│   ├── train.py                    # Script huấn luyện
│   ├── evaluate.py                 # Script đánh giá & so sánh 4 experiments
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/             # UI components (Board, StatsCard...)
│   │   ├── App.jsx                 # Layout và logic chính
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── weights/
│   └── colab/                      # File model (.pth) và history (.pkl)
│       ├── dqn_tetris_grouped_game_score.pth
│       ├── dqn_tetris_grouped_heuristic.pth
│       ├── dqn_tetris_ddqn_cur_grouped_game_score.pth
│       └── dqn_tetris_ddqn_cur_grouped_heuristic.pth
│
├── notebooks/                      # Jupyter Notebooks phân tích
│
├── .gitignore
└── README.md
```

---

## 🚀 Cài đặt & Khởi chạy

### 1. Backend (Python 3.11+)

```bash
# Tạo & kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Cài thư viện
pip install -r backend/requirements.txt

# Chạy API server
cd backend
uvicorn api.app:app --reload --port 8000
```

### 2. Frontend (Node.js 18+)

```bash
cd frontend
npm install
npm run dev
```

Truy cập dashboard tại **http://localhost:5173**

---

## 🧪 Huấn luyện & Đánh giá

### Huấn luyện

```bash
cd backend
python train.py
```

Config hyperparameters tại `backend/experiments/configs/colab_config.yaml`.

### Đánh giá 4 experiments

```bash
cd backend
python evaluate.py                 # 30 episodes mỗi model (mặc định)
python evaluate.py --episodes 50   # tuỳ chỉnh số episodes
```

Kết quả lưu tự động vào `backend/results/`:
| File | Nội dung |
|---|---|
| `*_summary.csv` | Bảng thống kê — mở bằng Excel / Google Sheets |
| `*_report.txt` | Báo cáo dạng bảng đọc trực tiếp trên terminal |
| `*_report.md` | Báo cáo Markdown — render được trên GitHub / VS Code |

---

## 🧠 Thiết kế DRL

### State (4 features)

| Feature | Mô tả |
|---|---|
| `lines_cleared` | Số dòng vừa xóa trong bước này |
| `holes` | Số ô trống bị chặn phía trên bởi ô đã lấp |
| `bumpiness` | Tổng chênh lệch chiều cao giữa các cột liền kề |
| `height` | Tổng chiều cao của tất cả các cột |

### Action Space (Grouped)

Mỗi action là một cặp `(x_pos, rotation)` — thả thẳng piece xuống vị trí đó (hard drop). AI chọn action tốt nhất từ tất cả vị trí khả dĩ trong một bước.

### Reward Functions

**Game Score:**
```
r = 1 + lines² × 10  (−2 nếu game over)
```

**Heuristic (Δf):**
```
f = −0.51·Height + 0.76·Lines − 0.36·Holes − 0.18·Bumpiness
r = f_new − f_prev   (−2 nếu game over)
```

---

## 📊 Kết quả Thí nghiệm (20 episodes/model)

| Model | Lines/ván | Pieces/ván | Bumpiness |
|---|---|---|---|
| **DDQN+Curriculum – Heuristic** ★ | **112.97** | **310.67** | 20.23 |
| DQN – Heuristic | 73.10 | 205.93 | 35.57 |
| DQN – Game Score | 20.73 | 82.37 | **4.50** ★ |
| DDQN+Curriculum – Game Score | 14.97 | 63.77 | 12.63 |

> **Nhận xét**: Heuristic reward dạy AI chơi hiệu quả hơn — xóa nhiều dòng hơn và sống lâu hơn. DDQN + Curriculum tiếp tục cải thiện thêm ~54% Lines so với DQN thuần.