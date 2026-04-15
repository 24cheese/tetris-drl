import numpy as np

# Định nghĩa 7 khối Tetromino cơ bản
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

class TetrisEngine:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int8)
        
        self.current_piece = None
        self.piece_x = 0
        self.piece_y = 0
        self.score = 0
        self.game_over = False

    def reset(self):
        """Khởi tạo lại bàn cờ và spawn khối đầu tiên"""
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.score = 0
        self.game_over = False
        self._spawn_piece()
        return self.board
    
    def _spawn_piece(self):
        """Chọn ngẫu nhiên 1 khối và đặt nó ở vị trí trên cùng giữa bàn cờ"""
        shape_name = np.random.choice(list(TETROMINOS.keys()))
        self.current_piece = np.array(TETROMINOS[shape_name], dtype=np.int8)
        
        # Đặt mảnh mới ở giữa trục X và trên cùng trục Y
        self.piece_x = self.width // 2 - self.current_piece.shape[1] // 2
        self.piece_y = 0

        # Nếu vừa đẻ ra đã va chạm -> Game Over
        if self._check_collision(self.current_piece, self.piece_x, self.piece_y):
            self.game_over = True

    def _rotate_piece(self):
        """Xoay ma trận 90 độ thuận chiều kim đồng hồ"""
        # Thuật toán xoay ma trận: Chuyển vị (Transpose) rồi lật (Flip)
        rotated = np.rot90(self.current_piece, k=-1)
        
        # Chỉ áp dụng nếu việc xoay không gây va chạm
        if not self._check_collision(rotated, self.piece_x, self.piece_y):
            self.current_piece = rotated

    def _check_collision(self, piece, x, y):
        """
        Thuật toán phát hiện va chạm siêu tốc:
        Duyệt qua các phần tử có giá trị '1' của khối và đối chiếu với bàn cờ.
        """
        h, w = piece.shape
        for i in range(h):
            for j in range(w):
                if piece[i, j] == 1:
                    board_x = x + j
                    board_y = y + i
                    
                    # Chạm tường (trái, phải, đáy)
                    if board_x < 0 or board_x >= self.width or board_y >= self.height:
                        return True
                    
                    # Chạm các khối khác trên bàn cờ (y >= 0 để bỏ qua viền trên)
                    if board_y >= 0 and self.board[board_y, board_x] != 0:
                        return True
        return False

    def _lock_piece_and_clear_lines(self):
        """Hàn khối hiện tại vào bàn cờ và thuật toán xóa dòng"""
        h, w = self.current_piece.shape
        for i in range(h):
            for j in range(w):
                if self.current_piece[i, j] == 1:
                    self.board[self.piece_y + i, self.piece_x + j] = 1

        # Lọc ra các dòng đã đầy (tất cả các ô đều bằng 1)
        # Sử dụng sức mạnh của numpy: kiểm tra dòng nào không có số 0
        full_lines = np.all(self.board != 0, axis=1)
        num_lines_cleared = np.sum(full_lines)

        if num_lines_cleared > 0:
            # Xóa các dòng đầy
            self.board = self.board[~full_lines]
            # Chèn thêm các dòng trống ở trên cùng
            empty_rows = np.zeros((num_lines_cleared, self.width), dtype=np.int8)
            self.board = np.vstack((empty_rows, self.board))
            
            # Cập nhật điểm (tạm thời)
            self.score += num_lines_cleared ** 2 * 10
            
        self._spawn_piece()
        return num_lines_cleared