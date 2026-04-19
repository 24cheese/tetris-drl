import React from 'react';

const Board = ({ boardMatrix }) => {
  if (!boardMatrix || boardMatrix.length === 0) {
    return (
      <div className="flex items-center justify-center w-[250px] h-[500px] bg-[#f6f5f4] border border-[rgba(0,0,0,0.1)] rounded-lg text-[#a39e98] font-medium">
        <div className="flex flex-col items-center gap-3">
          <div className="w-6 h-6 border-2 border-[#0075de]/20 border-t-[#0075de] rounded-full animate-spin"></div>
          <span className="text-sm">Loading grid...</span>
        </div>
      </div>
    );
  }

  const colorMap = {
    0: 'bg-transparent', 
    1: 'bg-[#2a9d99]', // Teal (Success/I)
    2: 'bg-[#0075de]', // Notion Blue (J)
    3: 'bg-[#dd5b00]', // Orange (L)
    4: 'bg-[#ffc107]', // Yellow (O)
    5: 'bg-[#1aae39]', // Green (S)
    6: 'bg-[#ff64c8]', // Pink (T)
    7: 'bg-[#eb5757]', // Red (Z)
  };

  return (
    <div className="relative p-2.5 bg-white border border-[rgba(0,0,0,0.1)] rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.05)]">
      <div className="grid grid-cols-10 gap-[1.5px] bg-[#e8e7e6] border-[1.5px] border-[#e8e7e6] overflow-hidden rounded-md shadow-inner">
        {boardMatrix.map((row, rowIndex) => (
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`w-5 h-5 sm:w-7 sm:h-7 transition-all duration-150 ${
                cell === 0 
                  ? 'bg-white'
                  : `${colorMap[cell]} rounded-[4px] shadow-[inset_0_1px_2px_rgba(255,255,255,0.2)]`
              }`}
            />
          ))
        ))}
      </div>
    </div>
  );
};

export default Board;