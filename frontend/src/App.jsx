import React, { useState, useEffect, useRef } from 'react';
import Board from './components/Board';
import { Play, Pause, SkipForward, RotateCcw, Box, Activity, Trophy, Info } from 'lucide-react';

function App() {
  const [gameState, setGameState] = useState({
    board: [],
    status: 'loading',
    info: { holes: 0, bumpiness: 0 }
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [lastAction, setLastAction] = useState('-');
  const [totalReward, setTotalReward] = useState(0);

  const intervalRef = useRef(null);

  const fetchGameStart = async () => {
    try {
      setIsPlaying(false);
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

  useEffect(() => {
    fetchGameStart();
  }, []);

  const fetchNextStep = async () => {
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
      if (data.status === 'game_over') setIsPlaying(false);

    } catch (error) {
      console.error("Lỗi khi gọi next-step:", error);
      setIsPlaying(false);
    }
  };

  useEffect(() => {
    if (isPlaying && gameState.status !== 'game_over') {
      intervalRef.current = setInterval(() => {
        fetchNextStep();
      }, 100);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isPlaying, gameState.status]);

  return (
    <div className="min-h-screen w-full overflow-auto bg-[#ffffff] flex flex-col items-center font-sans text-[rgba(0,0,0,0.95)] p-4 sm:p-8">
      <div className="max-w-5xl w-full flex items-center justify-between mb-8 pt-4 ">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[rgba(0,0,0,0.95)] rounded-lg flex items-center justify-center text-white shadow-sm">
            <Box size={24} strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight leading-none mb-1">Tetris AI</h1>
            <p className="text-[12px] text-[#615d59] font-medium uppercase tracking-wider">Deep Reinforcement Learning</p>
          </div>
        </div>
        <div className="hidden sm:flex gap-6">
          <a href="#" className="flex items-center gap-2 text-[14px] font-medium text-[rgba(0,0,0,0.6)] hover:text-[#0075de] transition-colors">
            Docs
          </a>
          <a href="https://github.com/24cheese/tetris-drl" className="flex items-center gap-2 text-[14px] font-medium text-[rgba(0,0,0,0.6)] hover:text-[#0075de] transition-colors">
            GitHub
          </a>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row items-start justify-center gap-12 max-w-5xl w-full">

        {/* LEFT COLUMN */}
        <div className="flex flex-col items-center w-full lg:w-auto">
          <div className="mb-10 text-center lg:text-left w-full">
            <h2 className="text-[36px] font-bold tracking-[-0.03em] leading-tight mb-3">Agent Training View</h2>
            <p className="text-[#615d59] text-[16px] max-w-md leading-relaxed">
              Observe how the neural network perceives and reacts to the game state in real-time.
            </p>
          </div>

          <div className="relative">
            {gameState.status === 'game_over' && (
              <div className="absolute -inset-3 border-2 border-[#eb5757] rounded-2xl z-20 game-over-pulse pointer-events-none"></div>
            )}

            <div className="bg-white notion-shadow rounded-2xl p-1 notion-border">
              <Board boardMatrix={gameState.board} />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="flex flex-col w-full lg:w-[400px] gap-8">

          {/* Main Stats Card */}
          <div className="bg-white notion-border rounded-2xl p-7 notion-shadow relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-5">
              <Activity size={80} />
            </div>

            <div className="flex items-center justify-between mb-10 border-b border-[rgba(0,0,0,0.05)] pb-5">
              <div className="flex flex-col">
                <span className="text-[11px] font-black text-[rgba(0,0,0,0.4)] uppercase tracking-[0.15em] mb-1">Status</span>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${gameState.status === 'playing' ? 'bg-[#1aae39] animate-pulse' : 'bg-[#eb5757]'}`}></div>
                  <span className="text-[14px] font-bold capitalize">{gameState.status.replace('_', ' ')}</span>
                </div>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[11px] font-black text-[rgba(0,0,0,0.4)] uppercase tracking-[0.15em] mb-1">Engine</span>
                <span className="text-[13px] font-bold text-[#615d59]">DQN v1.0</span>
              </div>
            </div>

            <div className="space-y-10">
              <div className="group">
                <span className="text-[12px] font-bold text-[#615d59] uppercase tracking-wider block mb-3 group-hover:text-[#0075de] transition-colors">Neural Output</span>
                <div className="bg-[#f6f5f4] p-4 rounded-xl font-mono text-[15px] font-bold text-[rgba(0,0,0,0.8)] border border-[rgba(0,0,0,0.05)] shadow-sm flex items-center gap-3">
                  <span className="text-[#0075de] font-black">▶</span> {lastAction}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-8">
                <div className="p-1">
                  <span className="text-[12px] font-bold text-[#615d59] uppercase tracking-wider block mb-2">Holes</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-[32px] font-black tracking-tight">{gameState.info.holes}</span>
                    <span className="text-[12px] text-[#a39e98] font-medium">units</span>
                  </div>
                </div>
                <div className="p-1">
                  <span className="text-[12px] font-bold text-[#615d59] uppercase tracking-wider block mb-2">Bumpiness</span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-[32px] font-black tracking-tight">{gameState.info.bumpiness}</span>
                    <span className="text-[12px] text-[#a39e98] font-medium">val</span>
                  </div>
                </div>
              </div>

              <div className="pt-6 border-t border-[rgba(0,0,0,0.05)]">
                <div className="flex items-center gap-2 mb-2">
                  <Trophy size={16} className="text-[#0075de]" />
                  <span className="text-[12px] font-bold text-[#615d59] uppercase tracking-wider">Accumulated Reward</span>
                </div>
                <div className="text-[48px] font-black tracking-[-0.05em] text-[#0075de] leading-none">
                  {totalReward.toFixed(1)}
                </div>
              </div>
            </div>
          </div>

          {/* Action Controls */}
          <div className="flex flex-col gap-4">
            <div className="flex gap-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={gameState.status === 'game_over'}
                className={`flex-[3] py-4 px-6 font-bold rounded-xl transition-all text-[15px] flex items-center justify-center gap-3 shadow-md hover:shadow-lg active:scale-[0.98] ${isPlaying
                    ? 'bg-white text-[rgba(0,0,0,0.8)] border border-[rgba(0,0,0,0.15)] hover:bg-[#f6f5f4]'
                    : 'bg-[#0075de] text-white border-transparent hover:bg-[#005bab] disabled:opacity-30'
                  }`}
              >
                {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" />}
                {isPlaying ? 'PAUSE AGENT' : 'START AUTO PLAY'}
              </button>

              <button
                onClick={fetchNextStep}
                disabled={isPlaying || gameState.status === 'game_over'}
                className="flex-1 bg-white border border-[rgba(0,0,0,0.15)] hover:bg-[#f6f5f4] text-[rgba(0,0,0,0.8)] rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98] flex items-center justify-center disabled:opacity-30"
                title="Next Step"
              >
                <SkipForward size={22} fill="currentColor" />
              </button>
            </div>

            <button
              onClick={fetchGameStart}
              className="w-full py-4 bg-transparent hover:bg-[rgba(0,0,0,0.03)] text-[#a39e98] hover:text-[rgba(0,0,0,0.7)] rounded-xl transition-all font-bold text-[13px] tracking-widest flex items-center justify-center gap-2 uppercase"
            >
              <RotateCcw size={14} /> Reset Training Environment
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;