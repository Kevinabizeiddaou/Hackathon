import React from 'react';
import { Play, Clock } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { formatTime } from '../utils/graphUtils';

export const ChunkTimeline: React.FC = () => {
  const { chunks, selectedChunkId, setSelectedChunkId, setIsModalOpen } = useAppStore();

  if (chunks.length === 0) {
    return (
      <div className="glass-card rounded-3xl p-8 text-center">
        <div className="space-y-4">
          <Clock className="w-12 h-12 text-gray-400 mx-auto" />
          <h3 className="text-xl font-medium text-gray-600">No chunks generated yet</h3>
          <p className="text-gray-500">Upload a video and click "Generate Chunks" to get started</p>
        </div>
      </div>
    );
  }

  const handleChunkClick = (chunkId: string) => {
    setSelectedChunkId(chunkId);
    setIsModalOpen(true);
  };

  return (
    <div className="glass-card rounded-3xl p-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Video Chunks</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {chunks.map((chunk) => (
          <div
            key={chunk.id}
            className={`chunk-card p-4 ${
              selectedChunkId === chunk.id ? 'selected' : ''
            }`}
            onClick={() => handleChunkClick(chunk.id)}
          >
            {/* Thumbnail */}
            <div className="relative mb-3">
              <div className="w-full h-32 bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl flex items-center justify-center">
                <Play className="w-8 h-8 text-blue-600" />
              </div>
              <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                #{chunk.id.slice(-3)}
              </div>
            </div>

            {/* Time Range */}
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Clock className="w-4 h-4" />
                <span>{formatTime(chunk.startMs)} – {formatTime(chunk.endMs)}</span>
              </div>
              
              {/* Analyze Button */}
              <button className="w-full mt-3 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 px-3 rounded-lg transition-all duration-200 hover:shadow-md">
                Analyze
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
