export type Chunk = {
  id: string;           // e.g., "c003"
  startMs: number;      // 4000
  endMs: number;        // 6000
  thumbnailUrl: string; // blob or server URL
};

export type ObjectItem = {
  name: string;                // "person"
  description?: string;        // "left dancer moving hips and hands"
};

export type PredictedAction = {
  id: string;
  source: string;
  target: string;
  action: string;
  probability: number;
};

export type AnalyzeResponse = {
  clip_id: string;
  video_meta: { fps?: number|null; duration_sec?: number|null; frames?: number|null };
  objects: ObjectItem[];
  relations: [string, string, string][]; // [subject, relation, object]
  predicted_actions?: PredictedAction[]; // k most probable next actions
  narrative: string;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  type: 'current' | 'predicted';
  probability?: number;
};

export type GraphData = {
  nodes: { id: string; label: string; desc?: string }[];
  edges: GraphEdge[];
};

export type AppState = {
  videoFile: File | null;
  chunks: Chunk[];
  selectedChunkId: string | null;
  analysisByChunkId: Record<string, AnalyzeResponse>;
  loadingAnalyze: boolean;
  mockMode: boolean;
  isModalOpen: boolean;
};
