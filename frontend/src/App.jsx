import React, { useState, useEffect, useRef } from 'react';
import Board from './components/Board';

function App() {
  const [gameState, setGameState] = useState({
    board: [],
    status: 'loading',
    info: { holes: 0, bumpiness: 0 }
  });
  
  // Các state mới cho việc điều khiển và hiển thị
  const [isPlaying, setIsPlaying] = useState(false);
  const [lastAction, setLastAction] = useState('-');
  const [totalReward, setTotalReward] = useState(0);

  // 1. Lấy trạng thái khởi đầu
  const fetchGameStart = async () => {
    try {
      setIsPlaying(false); // Dừng auto-play nếu đang chạy
      setLastAction('-');
      setTotalReward(0);
      
      const response = await fetch('http://127.0.0.1:8000/api/start');
      const data = await response.json();
      
      setGameState({
        board: data.board,
        status: data.status,
        info: data.info
      });
    } catch (error) {
      console.error("Lỗi kết nối Backend:", error);
    }
  };

  // Chạy 1 lần khi load trang
  useEffect(() => {
    fetchGameStart();
  }, []);

  // 2. Hàm gọi AI đi 1 bước
  const fetchNextStep = async () => {
    // Nếu game over thì không gọi nữa và tắt chế độ auto-play
    if (gameState.status === 'game_over') {
      setIsPlaying(false);
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:8000/api/next-step', {
        method: 'POST'
      });
      const data = await response.json();
      
      setGameState({
        board: data.board,
        status: data.status,
        info: data.info
      });
      
      if (data.action) setLastAction(data.action);
      if (data.reward) setTotalReward(prev => prev + data.reward);
      
      if (data.status === 'game_over') {
        setIsPlaying(false);
      }
      
    } catch (error) {
      console.error("Lỗi khi gọi next-step:", error);
      setIsPlaying(false);
    }
  };

  // 3. Vòng lặp Auto-Play sử dụng useEffect
  useEffect(() => {
    let intervalId;
    if (isPlaying && gameState.status !== 'game_over') {
      // Cứ mỗi 200ms (0.2s) sẽ tự động gọi fetchNextStep 1 lần
      intervalId = setInterval(() => {
        fetchNextStep();
      }, 200); 
    }
    
    // Dọn dẹp interval khi component unmount hoặc isPlaying thay đổi
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPlaying, gameState.status]);

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center font-sans text-gray-100">
      <div className="flex flex-col md:flex-row gap-8 items-start">
        
        {/* Cột trái: Bàn cờ */}
        <div>
          <h1 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
            Tetris DRL Agent
          </h1>
          <Board boardMatrix={gameState.board} />
          
          {/* Thông báo Game Over */}
          {gameState.status === 'game_over' && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-500 text-red-200 text-center rounded-lg animate-pulse">
              GAME OVER! AI đã chạm nóc.
            </div>
          )}
        </div>

        {/* Cột phải: Bảng điều khiển */}
        <div className="w-72 bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-xl">
          <h2 className="text-xl font-semibold mb-4 border-b border-gray-700 pb-2">Thông số AI</h2>
          
          <div className="space-y-4 mb-8">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Trạng thái:</span>
              <span className={`font-mono font-bold ${gameState.status === 'playing' || gameState.status === 'started' ? 'text-green-400' : 'text-red-400'}`}>
                {gameState.status}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Hành động vừa chọn:</span>
              <span className="font-mono text-cyan-400 bg-gray-800 px-2 py-1 rounded">{lastAction}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Tổng Reward:</span>
              <span className="font-mono text-yellow-400">{totalReward.toFixed(1)}</span>
            </div>
            
            <div className="h-px bg-gray-800 my-2"></div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Lỗ hổng (Holes):</span>
              <span className="font-mono text-xl">{gameState.info.holes}</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Mấp mô (Bumpiness):</span>
              <span className="font-mono text-xl">{gameState.info.bumpiness}</span>
            </div>
          </div>

          {/* Cụm nút điều khiển */}
          <div className="space-y-3">
            <div className="flex gap-2">
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={gameState.status === 'game_over'}
                className={`flex-1 py-2 font-bold rounded-lg transition-colors ${
                  isPlaying 
                    ? 'bg-red-600 hover:bg-red-700 text-white' 
                    : 'bg-green-600 hover:bg-green-700 text-white disabled:opacity-50'
                }`}
              >
                {isPlaying ? '⏸ Tạm dừng' : '▶ Tự động chơi'}
              </button>
              
              <button 
                onClick={fetchNextStep}
                disabled={isPlaying || gameState.status === 'game_over'}
                className="px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 transition-colors"
                title="Đi từng bước"
              >
                ⏭
              </button>
            </div>

            <button 
              onClick={fetchGameStart}
              className="w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-600 transition-colors"
            >
              🔄 Reset Game
            </button>
          </div>
          
        </div>

      </div>
    </div>
  );
}

export default App;