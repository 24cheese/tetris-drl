import React, { useState, useEffect, useRef } from 'react';
import Board from './components/Board';
import { Play, Pause, SkipForward, RotateCcw, Box, Activity, Trophy, Zap, Timer, Hash, Layers, TrendingUp } from 'lucide-react';

const API = 'http://127.0.0.1:8000';

// Định nghĩa 4 experiments
const EXPERIMENTS = [
  {
    key:         'game_score',
    label:       'DQN',
    sublabel:    'Game Score',
    color:       '#0075de',
    bg:          'rgba(0,117,222,0.08)',
    description: '1 + lines² × 10 − 2·GO',
    tag:         'DQN',
  },
  {
    key:         'heuristic',
    label:       'DQN',
    sublabel:    'Heuristic',
    color:       '#7c3aed',
    bg:          'rgba(124,58,237,0.08)',
    description: 'Δf = −0.51H + 0.76L − 0.36O − 0.18B',
    tag:         'DQN',
  },
  {
    key:         'ddqn_cur_game_score',
    label:       'DDQN+CUR',
    sublabel:    'Game Score',
    color:       '#ea5a0c',
    bg:          'rgba(234,90,12,0.08)',
    description: '1 + lines² × 10 − 2·GO',
    tag:         'DDQN',
  },
  {
    key:         'ddqn_cur_heuristic',
    label:       'DDQN+CUR',
    sublabel:    'Heuristic',
    color:       '#1aae39',
    bg:          'rgba(26,174,57,0.08)',
    description: 'Δf = −0.51H + 0.76L − 0.36O − 0.18B',
    tag:         'DDQN',
  },
];

function App() {
  const [gameState, setGameState] = useState({
    board: [],
    status: 'loading',
    info: { holes: 0, bumpiness: 0, height: 0, lines_cleared: 0 },
  });

  const [isPlaying, setIsPlaying]     = useState(false);
  const [lastAction, setLastAction]   = useState('—');
  const [totalReward, setTotalReward] = useState(0);
  const [modelKey, setModelKey]       = useState('game_score');
  const [switching, setSwitching]     = useState(false);
  const [steps, setSteps]             = useState(0);
  const [linesTotal, setLinesTotal]   = useState(0);
  const [elapsed, setElapsed]         = useState(0);

  const intervalRef = useRef(null);
  const timerRef    = useRef(null);
  const stepRef     = useRef(null); // tránh stale closure

  const activeExp = EXPERIMENTS.find(e => e.key === modelKey) || EXPERIMENTS[0];

  // Game Start
  const fetchGameStart = async () => {
    try {
      setIsPlaying(false);
      clearInterval(intervalRef.current);
      clearInterval(timerRef.current);
      setLastAction('—');
      setTotalReward(0);
      setSteps(0);
      setLinesTotal(0);
      setElapsed(0);

      const res  = await fetch(`${API}/api/start`);
      const data = await res.json();

      setGameState({
        board:  data.board,
        status: data.status,
        info:   data.info || { holes: 0, bumpiness: 0, height: 0, lines_cleared: 0 },
      });
      if (data.model_key) setModelKey(data.model_key);
    } catch (err) {
      console.error('Lỗi kết nối Backend:', err);
    }
  };

  useEffect(() => { fetchGameStart(); }, []);

  // Next Step
  const fetchNextStep = async () => {
    if (gameState.status === 'game_over') { setIsPlaying(false); return; }
    try {
      const res  = await fetch(`${API}/api/next-step`, { method: 'POST' });
      const data = await res.json();

      setGameState({
        board:  data.board,
        status: data.status,
        info:   data.info || { holes: 0, bumpiness: 0, height: 0, lines_cleared: 0 },
      });

      if (data.action) setLastAction(data.action);
      if (data.reward != null) setTotalReward(prev => prev + data.reward);
      if (data.steps  != null) setSteps(data.steps);
      if (data.lines  != null) setLinesTotal(data.lines);

      if (data.status === 'game_over') {
        setIsPlaying(false);
        clearInterval(timerRef.current);
      }
    } catch (err) {
      console.error('Lỗi gọi next-step:', err);
      setIsPlaying(false);
    }
  };

  stepRef.current = fetchNextStep;

  useEffect(() => {
    if (isPlaying && gameState.status !== 'game_over') {
      intervalRef.current = setInterval(() => stepRef.current(), 120);
      timerRef.current    = setInterval(() => setElapsed(e => e + 1), 1000);
    } else {
      clearInterval(intervalRef.current);
      clearInterval(timerRef.current);
    }
    return () => {
      clearInterval(intervalRef.current);
      clearInterval(timerRef.current);
    };
  }, [isPlaying, gameState.status]);

  // Switch Model
  const switchModel = async (key) => {
    if (key === modelKey || switching) return;
    setSwitching(true);
    setIsPlaying(false);
    try {
      const res  = await fetch(`${API}/api/switch-model?model_key=${key}`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setModelKey(data.model_key);
        setLastAction('—');
        setTotalReward(0);
        setSteps(0);
        setLinesTotal(0);
        setElapsed(0);
        const startRes  = await fetch(`${API}/api/start`);
        const startData = await startRes.json();
        setGameState({
          board:  startData.board,
          status: startData.status,
          info:   startData.info || { holes: 0, bumpiness: 0, height: 0, lines_cleared: 0 },
        });
      }
    } catch (err) {
      console.error('Lỗi switch model:', err);
    } finally {
      setSwitching(false);
    }
  };

  // Helpers
  const fmtTime = s => `${String(Math.floor(s / 60)).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}`;

  const isGameOver = gameState.status === 'game_over';

  // Render
  return (
    <div className="min-h-screen w-full bg-[#fafafa] flex flex-col items-center font-sans text-[rgba(0,0,0,0.9)] overflow-auto">

      {/* HEADER */}
      <div className="w-full border-b border-[rgba(0,0,0,0.08)] bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center text-white shadow-sm"
                 style={{ backgroundColor: activeExp.color }}>
              <Box size={20} strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-[15px] font-bold leading-tight">Tetris DRL</h1>
              <p className="text-[10px] text-[#999] font-semibold uppercase tracking-widest">
                Deep Reinforcement Learning · Colab v6
              </p>
            </div>
          </div>
          {/* Active experiment badge */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-bold border"
               style={{ color: activeExp.color, borderColor: `${activeExp.color}33`, backgroundColor: activeExp.bg }}>
            <Zap size={12} />
            {activeExp.label} · {activeExp.sublabel}
          </div>
        </div>
      </div>

      {/* BODY */}
      <div className="max-w-6xl w-full mx-auto px-4 sm:px-6 py-8 flex flex-col xl:flex-row gap-8 items-start justify-center">

        {/* LEFT — Tetris Board */}
        <div className="flex flex-col items-center flex-shrink-0">
          <div className="mb-5 text-center">
            <h2 className="text-[28px] sm:text-[34px] font-black tracking-[-0.03em] leading-tight mb-2">
              Agent Playback
            </h2>
            <p className="text-[#888] text-[14px] max-w-[280px]">
              Real-time board view - neural network decisions
            </p>
          </div>

          {/* Board wrapper */}
          <div className="relative">
            {isGameOver && (
              <div className="absolute -inset-2 rounded-2xl z-20 pointer-events-none border-2 border-[#eb5757] game-over-pulse" />
            )}
            {/* Game Over Overlay */}
            {isGameOver && (
              <div className="absolute inset-0 z-30 rounded-xl flex flex-col items-center justify-center bg-black/60 backdrop-blur-[2px]">
                <div className="text-white text-center px-6">
                  <div className="text-[40px] mb-1">💀</div>
                  <div className="text-[20px] font-black mb-1">Game Over</div>
                  <div className="text-[13px] text-white/70 mb-4">{steps} pieces · {linesTotal} lines</div>
                  <button
                    onClick={fetchGameStart}
                    className="px-5 py-2 bg-white text-black font-bold text-[13px] rounded-xl hover:bg-white/90 transition-all active:scale-95"
                  >
                    ↺ Restart
                  </button>
                </div>
              </div>
            )}
            <div className="bg-white border border-[rgba(0,0,0,0.1)] rounded-2xl p-2 shadow-[0_4px_24px_rgba(0,0,0,0.07)]">
              <Board boardMatrix={gameState.board} accentColor={activeExp.color} />
            </div>
          </div>

          {/* Controls dưới board */}
          <div className="mt-5 flex gap-3 w-full max-w-[280px]">
            <button
              onClick={() => setIsPlaying(p => !p)}
              disabled={isGameOver}
              className="flex-[3] py-3.5 font-bold rounded-xl transition-all text-[14px] flex items-center justify-center gap-2 shadow-md hover:shadow-lg active:scale-[0.98] disabled:opacity-30"
              style={!isPlaying ? { backgroundColor: activeExp.color, color: 'white' } : { backgroundColor: 'white', color: 'rgba(0,0,0,0.8)', border: '1px solid rgba(0,0,0,0.15)' }}
            >
              {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>

            <button
              onClick={fetchNextStep}
              disabled={isPlaying || isGameOver}
              className="flex-1 bg-white border border-[rgba(0,0,0,0.12)] hover:bg-[#f6f5f4] text-[rgba(0,0,0,0.7)] rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98] flex items-center justify-center disabled:opacity-30"
              title="Step"
            >
              <SkipForward size={20} />
            </button>

            <button
              onClick={fetchGameStart}
              className="flex-1 bg-white border border-[rgba(0,0,0,0.12)] hover:bg-[#f6f5f4] text-[rgba(0,0,0,0.7)] rounded-xl transition-all shadow-md hover:shadow-lg active:scale-[0.98] flex items-center justify-center"
              title="Reset"
            >
              <RotateCcw size={18} />
            </button>
          </div>
        </div>

        {/* RIGHT — Control Panel */}
        <div className="flex flex-col w-full xl:w-[420px] gap-5 flex-shrink-0">

          {/* ── Experiment Selector ── */}
          <div className="bg-white border border-[rgba(0,0,0,0.08)] rounded-2xl p-5 shadow-[0_2px_12px_rgba(0,0,0,0.05)]">
            <div className="flex items-center gap-2 mb-4">
              <Layers size={13} className="text-[#999]" />
              <span className="text-[11px] font-black text-[#999] uppercase tracking-[0.15em]">
                Experiment
              </span>
            </div>

            {/* 2×2 grid */}
            <div className="grid grid-cols-2 gap-2.5">
              {EXPERIMENTS.map(exp => {
                const isActive = modelKey === exp.key;
                return (
                  <button
                    key={exp.key}
                    onClick={() => switchModel(exp.key)}
                    disabled={switching}
                    className={`relative flex flex-col items-start p-3.5 rounded-xl border transition-all text-left ${
                      switching ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer hover:scale-[1.02] active:scale-[0.98]'
                    }`}
                    style={isActive
                      ? { borderColor: exp.color, backgroundColor: exp.bg, boxShadow: `0 0 0 1px ${exp.color}` }
                      : { borderColor: 'rgba(0,0,0,0.08)', backgroundColor: 'white' }
                    }
                  >
                    {/* Tag badge */}
                    <span className="text-[9px] font-black uppercase tracking-widest mb-1.5 px-1.5 py-0.5 rounded-md"
                          style={{ color: exp.color, backgroundColor: `${exp.color}18` }}>
                      {exp.tag}
                    </span>
                    <span className="text-[13px] font-bold leading-tight" style={{ color: isActive ? exp.color : 'rgba(0,0,0,0.8)' }}>
                      {exp.label}
                    </span>
                    <span className="text-[11px] font-semibold" style={{ color: isActive ? exp.color : '#888' }}>
                      {exp.sublabel}
                    </span>
                    {/* Active indicator */}
                    {isActive && (
                      <div className="absolute top-2.5 right-2.5 w-2 h-2 rounded-full animate-pulse"
                           style={{ backgroundColor: exp.color }} />
                    )}
                    {/* Loading indicator */}
                    {switching && modelKey !== exp.key && (
                      <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-white/60">
                        <div className="w-4 h-4 border-2 border-[#ccc] border-t-transparent rounded-full animate-spin" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Reward formula */}
            <div className="mt-3.5 px-3 py-2 rounded-lg text-[11px] font-mono text-center"
                 style={{ backgroundColor: activeExp.bg, color: activeExp.color }}>
              {activeExp.description}
            </div>
          </div>

          {/* ── Status + Neural Output ── */}
          <div className="bg-white border border-[rgba(0,0,0,0.08)] rounded-2xl p-5 shadow-[0_2px_12px_rgba(0,0,0,0.05)]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${gameState.status === 'playing' ? 'bg-[#1aae39] animate-pulse' : isGameOver ? 'bg-[#eb5757]' : 'bg-[#ccc]'}`} />
                <span className="text-[12px] font-bold capitalize text-[#555]">
                  {gameState.status.replace('_', ' ')}
                </span>
              </div>
              <span className="text-[11px] font-bold px-2 py-0.5 rounded-md"
                    style={{ color: activeExp.color, backgroundColor: activeExp.bg }}>
                {activeExp.label} · {activeExp.sublabel}
              </span>
            </div>

            {/* Last Action */}
            <div className="mb-1">
              <span className="text-[10px] font-black text-[#aaa] uppercase tracking-widest block mb-1.5">
                Neural Output
              </span>
              <div className="bg-[#f6f5f4] rounded-xl px-4 py-3 font-mono text-[14px] font-bold flex items-center gap-2.5">
                <span className="text-[18px]" style={{ color: activeExp.color }}>▶</span>
                <span className="text-[rgba(0,0,0,0.75)]">{lastAction}</span>
              </div>
            </div>
          </div>

          {/* ── Stats Grid ── */}
          <div className="bg-white border border-[rgba(0,0,0,0.08)] rounded-2xl p-5 shadow-[0_2px_12px_rgba(0,0,0,0.05)]">
            <div className="flex items-center gap-2 mb-4">
              <Activity size={13} className="text-[#999]" />
              <span className="text-[11px] font-black text-[#999] uppercase tracking-[0.15em]">
                Board Stats
              </span>
            </div>

            {/* 3-column stats */}
            <div className="grid grid-cols-3 gap-3 mb-4">
              <StatBox label="Lines" value={linesTotal} unit="cleared" color={activeExp.color} />
              <StatBox label="Holes" value={gameState.info.holes ?? 0} unit="cells" />
              <StatBox label="Height" value={gameState.info.height ?? 0} unit="rows" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <StatBox label="Bumpiness" value={gameState.info.bumpiness ?? 0} unit="val" />
              <StatBox label="Pieces" value={steps} unit="placed" />
            </div>
          </div>

          {/* ── Time + Reward ── */}
          <div className="bg-white border border-[rgba(0,0,0,0.08)] rounded-2xl p-5 shadow-[0_2px_12px_rgba(0,0,0,0.05)]">
            <div className="flex items-center justify-between">
              {/* Time */}
              <div>
                <div className="flex items-center gap-1.5 mb-1">
                  <Timer size={12} className="text-[#aaa]" />
                  <span className="text-[10px] font-black text-[#aaa] uppercase tracking-widest">Time</span>
                </div>
                <span className="text-[32px] font-black tracking-tight font-mono text-[rgba(0,0,0,0.85)]">
                  {fmtTime(elapsed)}
                </span>
              </div>

              {/* Divider */}
              <div className="w-px h-14 bg-[rgba(0,0,0,0.07)]" />

              {/* Reward */}
              <div className="text-right">
                <div className="flex items-center justify-end gap-1.5 mb-1">
                  <Trophy size={12} style={{ color: activeExp.color }} />
                  <span className="text-[10px] font-black text-[#aaa] uppercase tracking-widest">Reward</span>
                </div>
                <span className="text-[32px] font-black tracking-tight" style={{ color: activeExp.color }}>
                  {totalReward.toFixed(1)}
                </span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// Stat Box Component
function StatBox({ label, value, unit, color }) {
  return (
    <div className="bg-[#f9f9f9] border border-[rgba(0,0,0,0.06)] rounded-xl p-3">
      <span className="text-[10px] font-black text-[#aaa] uppercase tracking-widest block mb-1">{label}</span>
      <div className="flex items-baseline gap-1">
        <span className="text-[24px] font-black tracking-tight leading-none" style={color ? { color } : {}}>
          {value}
        </span>
        <span className="text-[10px] text-[#bbb] font-semibold">{unit}</span>
      </div>
    </div>
  );
}

export default App;