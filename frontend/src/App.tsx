import React, { useEffect } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { ChunkTimeline } from './components/ChunkTimeline';
import { GraphModal } from './components/GraphModal';
import { useAppStore } from './store/useAppStore';
import { ApiService } from './services/api';

function App() {
  const { 
    mockMode, 
    setChunks, 
    setAnalysis, 
    setLoadingAnalyze,
    chunks,
    selectedChunkId,
    analysisByChunkId 
  } = useAppStore();

  // Load mock data when in mock mode
  useEffect(() => {
    if (mockMode && chunks.length === 0) {
      setChunks(ApiService.getMockChunks());
    }
  }, [mockMode, chunks.length, setChunks]);

  // Auto-load analysis for selected chunk in mock mode
  useEffect(() => {
    if (mockMode && selectedChunkId && !analysisByChunkId[selectedChunkId]) {
      const mockAnalysis = ApiService.getMockAnalysis(selectedChunkId);
      if (mockAnalysis) {
        setAnalysis(selectedChunkId, mockAnalysis);
      }
    }
  }, [mockMode, selectedChunkId, analysisByChunkId, setAnalysis]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-white/20 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              Video Scene Graph Explorer
            </h1>
            <p className="text-lg text-gray-600">
              Upload a video → pick a chunk → see what's happening as a graph
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          <UploadPanel />
          <ChunkTimeline />
        </div>
      </main>

      {/* Graph Modal */}
      <GraphModal />

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t border-white/20 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              Built with React, TypeScript, TailwindCSS, and Cytoscape.js
            </p>
            <p className="text-xs mt-2 text-gray-500">
              Video Scene Graph Explorer • Hackathon Demo
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
