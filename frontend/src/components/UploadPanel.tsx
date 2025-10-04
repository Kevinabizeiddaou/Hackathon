import React, { useRef } from 'react';
import { Upload, FileVideo, ToggleLeft, ToggleRight, Loader2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { ApiService } from '../services/api';

export const UploadPanel: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { 
    videoFile, 
    mockMode, 
    setVideoFile, 
    setMockMode, 
    setChunks, 
    clearAll 
  } = useAppStore();
  const [isGenerating, setIsGenerating] = React.useState(false);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
    }
  };

  const handleGenerateChunks = async () => {
    if (!videoFile) return;
    
    setIsGenerating(true);
    try {
      let chunks;
      if (mockMode) {
        // Use mock data
        chunks = ApiService.getMockChunks();
      } else {
        // Call real API
        chunks = await ApiService.chunkVideo(videoFile);
      }
      setChunks(chunks);
    } catch (error) {
      console.error('Error generating chunks:', error);
      // Fallback to mock data
      setChunks(ApiService.getMockChunks());
    } finally {
      setIsGenerating(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="glass-card rounded-3xl p-8 mb-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Upload Video</h2>
        <button
          onClick={() => setMockMode(!mockMode)}
          className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${
            mockMode 
              ? 'bg-green-100 text-green-700 hover:bg-green-200' 
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {mockMode ? <ToggleRight className="w-4 h-4" /> : <ToggleLeft className="w-4 h-4" />}
          {mockMode ? 'Mock Mode' : 'Live Mode'}
        </button>
      </div>

      <div className="space-y-6">
        {/* File Upload Area */}
        <div
          className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
            videoFile 
              ? 'border-green-400 bg-green-50' 
              : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/mov,video/avi"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          {videoFile ? (
            <div className="space-y-3">
              <FileVideo className="w-12 h-12 text-green-600 mx-auto" />
              <div>
                <p className="text-lg font-medium text-gray-800">{videoFile.name}</p>
                <p className="text-sm text-gray-600">{formatFileSize(videoFile.size)}</p>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <Upload className="w-12 h-12 text-gray-400 mx-auto" />
              <div>
                <p className="text-lg font-medium text-gray-700">Click to upload video</p>
                <p className="text-sm text-gray-500">MP4, MOV, AVI supported</p>
              </div>
            </div>
          )}
        </div>

        {/* Generate Chunks Button */}
        <div className="text-center">
          <button
            onClick={handleGenerateChunks}
            disabled={!videoFile || isGenerating}
            className={`btn-primary ${
              !videoFile || isGenerating
                ? 'opacity-50 cursor-not-allowed' 
                : 'hover:shadow-xl hover:scale-105'
            }`}
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate Chunks'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
