import { create } from 'zustand';
import { AppState, Chunk, AnalyzeResponse } from '../types';

interface AppStore extends AppState {
  setVideoFile: (file: File | null) => void;
  setChunks: (chunks: Chunk[]) => void;
  setSelectedChunkId: (id: string | null) => void;
  setAnalysis: (chunkId: string, analysis: AnalyzeResponse) => void;
  setLoadingAnalyze: (loading: boolean) => void;
  setMockMode: (mock: boolean) => void;
  setIsModalOpen: (open: boolean) => void;
  clearAll: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  // Initial state
  videoFile: null,
  chunks: [],
  selectedChunkId: null,
  analysisByChunkId: {},
  loadingAnalyze: false,
  mockMode: true, // Start in mock mode for demo
  isModalOpen: false,

  // Actions
  setVideoFile: (file) => set({ videoFile: file }),
  setChunks: (chunks) => set({ chunks }),
  setSelectedChunkId: (id) => set({ selectedChunkId: id }),
  setAnalysis: (chunkId, analysis) => set((state) => ({
    analysisByChunkId: { ...state.analysisByChunkId, [chunkId]: analysis }
  })),
  setLoadingAnalyze: (loading) => set({ loadingAnalyze: loading }),
  setMockMode: (mock) => set({ mockMode: mock }),
  setIsModalOpen: (open) => set({ isModalOpen: open }),
  clearAll: () => set({
    videoFile: null,
    chunks: [],
    selectedChunkId: null,
    analysisByChunkId: {},
    loadingAnalyze: false,
    isModalOpen: false
  })
}));
