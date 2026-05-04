import gymnasium as gym
import numpy as np
from env.tetris_engine import TetrisEngine


class TetrisEnv(gym.Env):
    """
    STATE  (x_k) : [lines_cleared, holes, bumpiness, total_height]
    REWARD (r_k+1):
        game_score : 1 + lines²×10 − 2·gameover
        heuristic  : Δf, f = −0.51H + 0.76L − 0.36O − 0.18B
    """
    def __init__(self, render_mode=None, reward_type='game_score'):
        super().__init__()
        self.engine      = TetrisEngine()
        self.render_mode = render_mode
        self.reward_type = reward_type      # "game_score" | "heuristic"
        self._prev_f     = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        stats        = self._get_heuristic_stats()
        self._prev_f = self._heuristic_f(0, stats)
        state        = [0, stats['holes'], stats['bumpiness'], stats['height']]
        return state, stats

    def step(self, action):
        """Grouped action: action = (x_pos, rotation)"""
        x, rotation = action
        self.engine.current_piece.rotation = rotation
        self.engine.current_piece.x        = x

        # Hard drop — thả thẳng xuống đáy
        while not self.engine._check_collision(self.engine.current_piece, 0, 1):
            self.engine.current_piece.y += 1

        self.engine._lock_piece()
        lines_cleared = self.engine._clear_lines()

        stats  = self._get_heuristic_stats()
        stats['lines_cleared'] = lines_cleared          # ← fix: expose cho info
        reward = self._calculate_reward(lines_cleared, stats)
        self.engine.game_over = self.engine._check_game_over()

        state = [lines_cleared, stats['holes'], stats['bumpiness'], stats['height']]
        return state, reward, self.engine.game_over, False, stats

    def get_possible_states(self):
        """Mô phỏng mọi vị trí thả gạch để AI chấm điểm"""
        possible     = {}
        orig_board   = self.engine.board.copy()
        orig_piece   = self.engine.current_piece.copy()

        for rotation in range(len(orig_piece.shapes)):
            for x in range(10):
                self.engine.current_piece          = orig_piece.copy()
                self.engine.current_piece.rotation = rotation
                self.engine.current_piece.x        = x
                self.engine.current_piece.y        = 0

                if self.engine._check_collision(self.engine.current_piece, 0, 0):
                    continue

                while not self.engine._check_collision(self.engine.current_piece, 0, 1):
                    self.engine.current_piece.y += 1

                self.engine._lock_piece()
                lines = self.engine._clear_lines()
                stats = self._get_heuristic_stats()

                possible[(x, rotation)] = [lines, stats['holes'], stats['bumpiness'], stats['height']]

                self.engine.board        = orig_board.copy()

        self.engine.current_piece = orig_piece
        return possible

    # Reward Functions

    def _calculate_reward(self, lines_cleared, stats):
        if self.reward_type == 'heuristic':
            # r = Δf, f = −0.51H + 0.76L − 0.36O − 0.18B
            f_new        = self._heuristic_f(lines_cleared, stats)
            reward       = f_new - self._prev_f
            self._prev_f = f_new
            if self.engine.game_over:
                reward -= 2.0
        else:
            # Game Score (v6): 1 + lines²×10 − 2·gameover
            reward = 1 + (lines_cleared ** 2) * 10
            if self.engine.game_over:
                reward -= 2.0
        return float(reward)

    def _heuristic_f(self, lines, stats):
        """f = −0.51·H + 0.76·L − 0.36·O − 0.18·B"""
        return (
            -0.51 * stats['height']
            + 0.76 * lines
            - 0.36 * stats['holes']
            - 0.18 * stats['bumpiness']
        )

    # Feature Extraction

    def _get_heuristic_stats(self):
        """Trích xuất 4 features từ board hiện tại."""
        board = self.engine.board

        # Holes: ô trống nằm dưới ô đã lấp
        holes = 0
        for col in board.T:
            indices = np.where(col != 0)[0]
            if len(indices) > 0:
                top = indices[0]
                holes += np.sum(col[top:] == 0)

        # Height của từng cột
        mask     = board != 0
        inv_h    = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 20)
        heights  = 20 - inv_h
        total_h  = int(np.sum(heights))

        # Bumpiness: tổng |h[i] − h[i+1]|
        bumpiness = int(np.sum(np.abs(np.diff(heights))))

        return {'holes': holes, 'bumpiness': bumpiness, 'height': total_h}

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