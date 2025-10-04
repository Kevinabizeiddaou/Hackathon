import type { Chunk, AnalyzeResponse, ObjectItem } from "../types";
import { mockChunks, mockAnalyzeById } from "../data/mockData";

const API_BASE_URL = "http://localhost:8000"; // FastAPI backend

export class ApiService {
  static async chunkVideo(videoFile: File): Promise<Chunk[]> {
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch(`${API_BASE_URL}/chunk`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Chunk API error:", error);
      // Fallback to mock data
      return mockChunks;
    }
  }

  static async analyzeChunk(
    videoFile: File,
    objects: ObjectItem[],
    useLLM: boolean = false
  ): Promise<AnalyzeResponse> {
    const formData = new FormData();
    formData.append("videoclip", videoFile);
    formData.append("objects_json", JSON.stringify(objects));
    formData.append("use_llm", useLLM.toString());

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Analyze API error:", error);
      throw error;
    }
  }

  static getMockAnalysis(chunkId: string): AnalyzeResponse | null {
    return mockAnalyzeById[chunkId] || null;
  }

  static getMockChunks(): Chunk[] {
    return mockChunks;
  }
}
