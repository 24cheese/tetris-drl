from tetris_env import TetrisEnv
import time
import os

def clear_console():
    """Hàm xóa màn hình terminal để tạo hiệu ứng animation"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # 1. Khởi tạo môi trường
    env = TetrisEnv(render_mode='human')
    state, info = env.reset()

    print("Bắt đầu test môi trường với Random Agent (Hành động ngẫu nhiên)...")
    time.sleep(2)

    # 2. Cho AI chơi thử tối đa 500 bước
    for step in range(500):
        clear_console()
        print(f"--- Bước {step+1} ---")

        # In ra bàn cờ hiện tại (Thay vì in số 0, 1, ta in ký tự cho dễ nhìn)
        board_state = env._get_state()
        for row in board_state:
            # Hiện '[]' cho gạch, '.' cho ô trống
            print(" ".join(["[]" if cell == 1 else " ." for cell in row]))

        # 3. Chọn ngẫu nhiên 1 hành động từ Action Space (0: Trái, 1: Phải, 2: Xoay, 3: Hard Drop, 4: Rơi tự nhiên)
        action = env.action_space.sample()
        
        action_names = {0: "Trái", 1: "Phải", 2: "Xoay", 3: "Thả mạnh (Hard Drop)", 4: "Rơi (No-op)"}
        print(f"\nHành động AI vừa chọn: {action_names[action]}")

        # 4. Truyền hành động vào môi trường
        state, reward, done, truncated, info = env.step(action)

        # In ra các thông số để test Reward Shaping
        print(f"Reward nhận được: {reward:.2f}")
        print(f"Số lỗ hổng (Holes): {info['holes']} | Độ mấp mô (Bumpiness): {info['bumpiness']}")

        time.sleep(0.15) # Tạm dừng 0.15 giây để mắt người kịp nhìn thấy hiệu ứng rơi

        # 5. Kiểm tra điều kiện thua
        if done:
            print("\nGame Over! AI đã xếp chạm nóc.")
            break

    env.close()

if __name__ == "__main__":
    main()