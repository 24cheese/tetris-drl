import numpy as np

TETROMINOS = {
    'I': [[1, 1, 1, 1]],
    'J': [[1, 0, 0], 
          [1, 1, 1]],
    'L': [[0, 0, 1], 
          [1, 1, 1]],
    'O': [[1, 1], 
          [1, 1]],
    'S': [[0, 1, 1], 
          [1, 1, 0]],
    'T': [[0, 1, 0], 
          [1, 1, 1]],
    'Z': [[1, 1, 0], 
          [0, 1, 1]]
}

class Piece:
    """Đóng gói khối gạch thành một Thực thể (Object) để AI dễ mô phỏng"""
    def __init__(self, shape_name):
        self.name = shape_name
        
        color_map = {'I': 1, 'J': 2, 'L': 3, 'O': 4, 'S': 5, 'T': 6, 'Z': 7}
        self.color_id = color_map[shape_name]
        
        base_shape = np.array(TETROMINOS[shape_name], dtype=np.int8) * self.color_id   
             
        all_rotations = [base_shape]
        for _ in range(3):
            all_rotations.append(np.rot90(all_rotations[-1], k=-1))
            
        self.shapes = []
        for s in all_rotations:
            if not any(np.array_equal(s, us) for us in self.shapes):
                self.shapes.append(s)

        self.rotation = 0
        self.x = 0
        self.y = 0

    @property
    def current_shape(self):
        return self.shapes[self.rotation % len(self.shapes)]

    def copy(self):
        p = Piece(self.name)
        p.rotation = self.rotation
        p.x = self.x
        p.y = self.y
        p.shapes = self.shapes
        return p


class TetrisEngine:
    """Lõi Game Engine đã được tinh chỉnh cho DRL Agent"""
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int8)
        self.current_piece = None
        self.game_over = False

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.game_over = False
        self._spawn_piece()

    def _spawn_piece(self):
        shape_name = np.random.choice(list(TETROMINOS.keys()))
        self.current_piece = Piece(shape_name)
        
        self.current_piece.x = self.width // 2 - self.current_piece.current_shape.shape[1] // 2
        self.current_piece.y = 0

        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True

    def _check_collision(self, piece, dx=0, dy=0):
        """Kiểm tra va chạm dựa vào tọa độ ảo (dx, dy) tính từ vị trí hiện tại của khối"""
        shape = piece.shapes[piece.rotation]
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j] != 0:
                    board_x = piece.x + j + dx
                    board_y = piece.y + i + dy
                    
                    if board_x < 0 or board_x >= self.width or board_y >= self.height:
                        return True
                    
                    if board_y >= 0 and self.board[board_y, board_x] != 0:
                        return True
        return False

    def _lock_piece(self):
        """Hàn khối gạch vào bàn cờ"""
        shape = self.current_piece.shapes[self.current_piece.rotation]
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j] !=0:
                    if self.current_piece.y + i >= 0:
                        self.board[self.current_piece.y + i, self.current_piece.x + j] = shape[i, j] 
                        
    def _clear_lines(self):
        """Xóa dòng và trả về số dòng ăn được"""
        full_lines = np.all(self.board != 0, axis=1)
        num_lines_cleared = np.sum(full_lines)

        if num_lines_cleared > 0:
            self.board = self.board[~full_lines]
            empty_rows = np.zeros((num_lines_cleared, self.width), dtype=np.int8)
            self.board = np.vstack((empty_rows, self.board))
            
        self._spawn_piece()
        return num_lines_cleared

    def _check_game_over(self):
        return self.game_over