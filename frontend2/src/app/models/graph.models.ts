export interface GraphNode {
  id: string;
  label: string;
  description: string;
  x?: number;
  y?: number;
  type: 'object' | 'person' | 'location' | 'other';
  confidence: number;
  boundingBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  type: 'current' | 'predicted';
  confidence: number;
  description?: string;
}

export interface VideoChunk {
  id: string;
  startTime: number;
  endTime: number;
  duration: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: string;
  frameUrl?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface PredictionContext {
  previousChunks: VideoChunk[];
  currentChunk: VideoChunk;
  actionContext: string;
  predictions: {
    event: string;
    probability: number;
    description: string;
  }[];
}
