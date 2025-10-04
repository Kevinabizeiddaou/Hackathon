import type { AnalyzeResponse, GraphData, GraphEdge } from '../types';

export function toGraph(data: AnalyzeResponse): GraphData {
  const uniq = <T,>(arr: T[]) => Array.from(new Set(arr.map(x => JSON.stringify(x)))).map(x => JSON.parse(x));
  
  const nodes = uniq(
    data.objects.map((o, i) => ({
      id: (o.name + "_" + i).toLowerCase(), // ensure uniqueness if multiple "person"
      label: o.name,
      desc: o.description || ""
    }))
  );

  // Current relations (solid edges)
  const currentEdges: GraphEdge[] = data.relations.map((r, i) => {
    const [subj, rel, obj] = r;
    const src = nodes.find(n => n.label.toLowerCase() === subj.toLowerCase())?.id || subj;
    const dst = nodes.find(n => n.label.toLowerCase() === obj.toLowerCase())?.id || obj;
    return { 
      id: "current_" + i, 
      source: src, 
      target: dst, 
      label: rel,
      type: 'current'
    };
  });

  // Predicted actions (dotted edges)
  const predictedEdges: GraphEdge[] = (data.predicted_actions || []).map((pred, i) => {
    const src = nodes.find(n => n.label.toLowerCase() === pred.source.toLowerCase())?.id || pred.source;
    const dst = nodes.find(n => n.label.toLowerCase() === pred.target.toLowerCase())?.id || pred.target;
    return {
      id: "predicted_" + i,
      source: src,
      target: dst,
      label: pred.action,
      type: 'predicted',
      probability: pred.probability
    };
  });

  const edges = [...currentEdges, ...predictedEdges];

  return { nodes, edges };
}

export function formatTime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

export function generateThumbnail(videoFile: File, timeMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      reject(new Error('Could not get canvas context'));
      return;
    }

    video.addEventListener('loadeddata', () => {
      video.currentTime = timeMs / 1000;
    });

    video.addEventListener('seeked', () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      resolve(canvas.toDataURL('image/jpeg', 0.8));
    });

    video.addEventListener('error', reject);
    video.src = URL.createObjectURL(videoFile);
  });
}
