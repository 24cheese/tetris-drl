import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.tetris_engine import TetrisEngine # Import Engine từ Bước 1

class TetrisEnv(gym.Env):
    """
    Môi trường Tetris chuẩn OpenAI Gymnasium với Action Space rời rạc (Bấm phím).
    """
    metadata = {'render_modes': ['human', 'api']}

    def __init__(self, render_mode=None):
        super(TetrisEnv, self).__init__()
        self.render_mode = render_mode
        self.engine = TetrisEngine(width=10, height=20)
        
        # 1. KHÔNG GIAN HÀNH ĐỘNG (5 Phím bấm)
        # 0: Trái, 1: Phải, 2: Xoay, 3: Hard Drop (Rơi cắm thẳng), 4: No-op (Chờ trọng lực)
        self.action_space = spaces.Discrete(5)
        
        # 2. KHÔNG GIAN TRẠNG THÁI (Ma trận 20x10)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.engine.height, self.engine.width), 
            dtype=np.int8
        )

        # Lưu lại heuristic của bước trước để so sánh Reward Shaping
        self.prev_heuristic = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        board = self.engine.reset()
        self.prev_heuristic = self._get_heuristic_stats()
        return self._get_state(), {}

    def step(self, action):
        """Thực thi 1 hành động và tính toán phần thưởng"""
        lines_cleared = 0
        piece_locked = False

        # --- 1. XỬ LÝ HÀNH ĐỘNG CỦA AI ---
        if action == 0: # Dịch Trái
            if not self.engine._check_collision(self.engine.current_piece, self.engine.piece_x - 1, self.engine.piece_y):
                self.engine.piece_x -= 1
        elif action == 1: # Dịch Phải
            if not self.engine._check_collision(self.engine.current_piece, self.engine.piece_x + 1, self.engine.piece_y):
                self.engine.piece_x += 1
        elif action == 2: # Xoay
            self.engine._rotate_piece()
        elif action == 3: # Hard Drop (Rơi chạm đáy ngay lập tức)
            while not self.engine._check_collision(self.engine.current_piece, self.engine.piece_x, self.engine.piece_y + 1):
                self.engine.piece_y += 1
            lines_cleared = self.engine._lock_piece_and_clear_lines()
            piece_locked = True
        # action == 4 là không làm gì, để mặc gạch tự rơi theo trọng lực ở bước 2

        # --- 2. ÁP DỤNG TRỌNG LỰC (Nếu chưa Hard Drop) ---
        if not piece_locked:
            # Mỗi nhịp (step), viên gạch tự động rơi xuống 1 ô
            if not self.engine._check_collision(self.engine.current_piece, self.engine.piece_x, self.engine.piece_y + 1):
                self.engine.piece_y += 1
            else:
                # Nếu không thể rơi thêm -> Chạm đáy -> Hàn khối
                lines_cleared = self.engine._lock_piece_and_clear_lines()
                piece_locked = True

        # --- 3. TÍNH TOÁN REWARD SHAPING (Cực kỳ quan trọng) ---
        reward = self._calculate_reward(lines_cleared, piece_locked)
        
        terminated = self.engine.game_over
        truncated = False
        info = self._get_heuristic_stats()

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        """Hợp nhất bàn cờ và viên gạch đang rơi thành 1 ma trận duy nhất để đưa vào AI"""
        state = np.copy(self.engine.board)
        if self.engine.current_piece is not None and not self.engine.game_over:
            h, w = self.engine.current_piece.shape
            for i in range(h):
                for j in range(w):
                    if self.engine.current_piece[i, j] == 1:
                        # Kiểm tra giới hạn để không bị lỗi index khi gạch vừa spawn
                        if 0 <= self.engine.piece_y + i < self.engine.height:
                            state[self.engine.piece_y + i, self.engine.piece_x + j] = 1
        return state

    def _calculate_reward(self, lines_cleared, piece_locked):
        """
        Hàm nhào nặn phần thưởng (Reward Shaping).
        Thay vì đợi ăn dòng mới thưởng, ta phạt/thưởng AI dựa trên cấu trúc bàn cờ.
        """
        if self.engine.game_over:
            return -100.0 # Phạt cực nặng khi thua

        reward = 0.0
        
        # 1. Thưởng cực lớn khi ăn dòng (Mục tiêu chính)
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 100 

        # 2. Đánh giá Heuristic khi một viên gạch vừa được đặt xuống
        if piece_locked:
            current_heuristic = self._get_heuristic_stats()
            
            # Tính toán sự chênh lệch (Delta) giữa bàn cờ mới và cũ
            delta_holes = current_heuristic['holes'] - self.prev_heuristic['holes']
            delta_bumpiness = current_heuristic['bumpiness'] - self.prev_heuristic['bumpiness']
            
            # Phạt nếu AI tạo thêm lỗ hổng (Lỗi chí mạng nhất trong Tetris)
            if delta_holes > 0:
                reward -= delta_holes * 50.0
                
            # Phạt nhẹ nếu làm bàn cờ gồ ghề hơn
            if delta_bumpiness > 0:
                reward -= delta_bumpiness * 10.0
                
            # Thưởng nhẹ nhỏ gọn gọn để khuyến khích đặt gạch an toàn (sống sót)
            reward += 10.0 
            
            # Cập nhật lại heuristic cho viên gạch tiếp theo
            self.prev_heuristic = current_heuristic
            
        else:
            # Khuyến khích AI nhấn Hard Drop để tăng tốc độ chơi, tránh rề rà
            reward += 0.1 

        return reward

    def _get_heuristic_stats(self):
        """Tính toán các chỉ số Heuristic chuyên gia từ bàn cờ hiện tại"""
        board = self.engine.board
        
        # 1. Số lỗ hổng (Holes): Các ô trống có khối gạch đè lên đầu
        holes = 0
        for col in range(self.engine.width):
            block_found = False
            for row in range(self.engine.height):
                if board[row, col] == 1:
                    block_found = True
                elif block_found and board[row, col] == 0:
                    holes += 1
                    
        # 2. Độ cao từng cột và độ mấp mô (Bumpiness)
        heights = np.zeros(self.engine.width)
        for col in range(self.engine.width):
            for row in range(self.engine.height):
                if board[row, col] == 1:
                    heights[col] = self.engine.height - row
                    break
                    
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
        
        return {
            "holes": holes,
            "bumpiness": bumpiness,
            "aggregate_height": np.sum(heights)
        }