import React from 'react';

const Board = ({ boardMatrix }) => {
  // Nếu chưa có dữ liệu từ API, hiển thị khung chờ
  if (!boardMatrix || boardMatrix.length === 0) {
    return (
      <div className="flex items-center justify-center w-[320px] h-[640px] bg-gray-900 border-4 border-gray-700 text-white">
        Đang tải bàn cờ...
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-1 border-4 border-gray-900 rounded-lg shadow-2xl inline-block">
      {/* Lưới Grid: 10 cột, mỗi ô cách nhau 1px (gap-px) để tạo viền */}
      <div className="grid grid-cols-10 gap-[1px] bg-gray-700">
        {boardMatrix.map((row, rowIndex) => (
          row.map((cell, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`w-6 h-6 sm:w-8 sm:h-8 transition-colors duration-75 ${
                cell === 1 
                  ? 'bg-cyan-500 shadow-[inset_0_0_8px_rgba(0,0,0,0.3)]' // Màu gạch (Cyan)
                  : 'bg-gray-900' // Màu nền ô trống
              }`}
            />
          ))
        ))}
      </div>
    </div>
  );
};

export default Board;