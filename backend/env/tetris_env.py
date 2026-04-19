import gymnasium as gym
import numpy as np
from env.tetris_engine import TetrisEngine

class TetrisEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.engine = TetrisEngine()
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        stats = self._get_heuristic_stats()
        # Trả về 4 features: [Lines, Holes, Bumpiness, Height]
        state = [0, stats['holes'], stats['bumpiness'], stats['height']]
        return state, stats

    def step(self, action):
        """
        action: tuple (x, rotation)
        """
        x, rotation = action
        
        self.engine.current_piece.rotation = rotation
        self.engine.current_piece.x = x
        
        # Thả rơi tự do xuống đáy
        while not self.engine._check_collision(self.engine.current_piece, 0, 1):
            self.engine.current_piece.y += 1
            
        self.engine._lock_piece()
        lines_cleared = self.engine._clear_lines()
        
        stats = self._get_heuristic_stats()
        reward = self._calculate_reward(lines_cleared, stats)
        self.engine.game_over = self.engine._check_game_over()
        
        state = [lines_cleared, stats['holes'], stats['bumpiness'], stats['height']]
        return state, reward, self.engine.game_over, False, stats

    def get_possible_states(self):
        """Mô phỏng mọi vị trí thả gạch để AI chấm điểm"""
        possible_states = {}
        original_board = self.engine.board.copy()
        original_piece = self.engine.current_piece.copy()
        
        for rotation in range(len(original_piece.shapes)):
            for x in range(10):
                self.engine.current_piece = original_piece.copy()
                self.engine.current_piece.rotation = rotation
                self.engine.current_piece.x = x
                self.engine.current_piece.y = 0
                
                if self.engine._check_collision(self.engine.current_piece, 0, 0):
                    continue
                    
                while not self.engine._check_collision(self.engine.current_piece, 0, 1):
                    self.engine.current_piece.y += 1
                    
                self.engine._lock_piece()
                lines = self.engine._clear_lines() 
                stats = self._get_heuristic_stats()
                
                feature_vector = [lines, stats['holes'], stats['bumpiness'], stats['height']]
                possible_states[(x, rotation)] = feature_vector
                
                self.engine.board = original_board.copy()
                
        self.engine.current_piece = original_piece
        return possible_states

    def _calculate_reward(self, lines_cleared, stats):
        """
        Phần thưởng chuẩn mực của DQN Tetris:
        Chỉ thưởng sinh tồn và ăn dòng. Phạt khi chết.
        Không trừ điểm lỗ hổng thủ công nữa!
        """
        # 1. Thưởng 1 điểm vì đã sống sót thêm 1 bước
        # 2. Thưởng đậm (hàm mũ) nếu ăn được dòng
        reward = 1.0 + (lines_cleared ** 2) * 10
        
        # 3. Phạt nếu bước đi này dẫn đến Game Over
        if self.engine.game_over:
            reward -= 2.0
            
        return float(reward)

    def _get_heuristic_stats(self):
        """Hàm trích xuất đặc trưng siêu tốc sử dụng Numpy"""
        board = self.engine.board
        
        holes = 0
        for col in board.T:
            indices = np.where(col != 0)[0]
            if len(indices) > 0:
                top = indices[0]
                holes += np.sum(col[top:] == 0)

        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
        heights = 20 - invert_heights
        
        total_height = int(np.sum(heights))
        
        # Độ mấp mô = Tổng trị tuyệt đối hiệu số chiều cao 2 cột liền kề
        diffs = np.abs(heights[:-1] - heights[1:])
        bumpiness = int(np.sum(diffs))
            
        return {
            'holes': holes,
            'bumpiness': bumpiness,
            'height': total_height
        }

    def _get_state(self):
        board_copy = self.engine.board.copy()
        if self.engine.current_piece:
            piece = self.engine.current_piece
            shape = piece.shapes[piece.rotation]
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell != 0 and 0 <= piece.y + y < 20 and 0 <= piece.x + x < 10:
                        board_copy[piece.y + y, piece.x + x] = cell 
        return board_copy

    def render(self):
        return self._get_state().tolist()