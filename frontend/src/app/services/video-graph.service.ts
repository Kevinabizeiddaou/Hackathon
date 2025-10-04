import { Injectable } from '@angular/core';
import { Observable, BehaviorSubject } from 'rxjs';
import { VideoChunk, GraphNode, GraphEdge, PredictionContext } from '../models/graph.models';

@Injectable({
  providedIn: 'root'
})
export class VideoGraphService {
  private videoChunksSubject = new BehaviorSubject<VideoChunk[]>([]);
  public videoChunks$ = this.videoChunksSubject.asObservable();

  private currentChunkSubject = new BehaviorSubject<VideoChunk | null>(null);
  public currentChunk$ = this.currentChunkSubject.asObservable();

  constructor() {}

  // Mock data for development and testing
  getMockVideoChunks(): VideoChunk[] {
    return [
      {
        id: 'chunk_1',
        startTime: 0,
        endTime: 5,
        duration: 5,
        summary: 'A person enters the kitchen and approaches the refrigerator.',
        nodes: [
          {
            id: 'person_1',
            label: 'Person',
            description: 'Adult person wearing casual clothes',
            type: 'person',
            confidence: 0.95,
            x: 200,
            y: 200
          },
          {
            id: 'refrigerator_1',
            label: 'Refrigerator',
            description: 'Large white refrigerator in the kitchen',
            type: 'object',
            confidence: 0.92,
            x: 400,
            y: 200
          },
          {
            id: 'kitchen_1',
            label: 'Kitchen',
            description: 'Modern kitchen with white cabinets',
            type: 'location',
            confidence: 0.88,
            x: 300,
            y: 100
          }
        ],
        edges: [
          {
            id: 'edge_1',
            source: 'person_1',
            target: 'kitchen_1',
            label: 'located in',
            type: 'current',
            confidence: 0.9
          },
          {
            id: 'edge_2',
            source: 'person_1',
            target: 'refrigerator_1',
            label: 'will open',
            type: 'predicted',
            confidence: 0.85,
            description: 'Person is likely to open the refrigerator based on movement pattern'
          }
        ]
      },
      {
        id: 'chunk_2',
        startTime: 5,
        endTime: 10,
        duration: 5,
        summary: 'Person opens the refrigerator and takes out a milk carton.',
        nodes: [
          {
            id: 'person_1',
            label: 'Person',
            description: 'Adult person reaching into refrigerator',
            type: 'person',
            confidence: 0.96,
            x: 200,
            y: 200
          },
          {
            id: 'refrigerator_1',
            label: 'Refrigerator',
            description: 'Open refrigerator with visible contents',
            type: 'object',
            confidence: 0.94,
            x: 400,
            y: 200
          },
          {
            id: 'milk_1',
            label: 'Milk',
            description: 'White milk carton being held',
            type: 'object',
            confidence: 0.89,
            x: 250,
            y: 250
          },
          {
            id: 'kitchen_1',
            label: 'Kitchen',
            description: 'Modern kitchen with white cabinets',
            type: 'location',
            confidence: 0.88,
            x: 300,
            y: 100
          }
        ],
        edges: [
          {
            id: 'edge_1',
            source: 'person_1',
            target: 'refrigerator_1',
            label: 'opening',
            type: 'current',
            confidence: 0.95
          },
          {
            id: 'edge_2',
            source: 'person_1',
            target: 'milk_1',
            label: 'holding',
            type: 'current',
            confidence: 0.92
          },
          {
            id: 'edge_3',
            source: 'person_1',
            target: 'kitchen_1',
            label: 'will move to counter',
            type: 'predicted',
            confidence: 0.78,
            description: 'Person will likely move to counter to pour milk'
          }
        ]
      },
      {
        id: 'chunk_3',
        startTime: 10,
        endTime: 15,
        duration: 5,
        summary: 'Person closes refrigerator and moves to the counter with milk.',
        nodes: [
          {
            id: 'person_1',
            label: 'Person',
            description: 'Adult person at kitchen counter',
            type: 'person',
            confidence: 0.94,
            x: 150,
            y: 300
          },
          {
            id: 'refrigerator_1',
            label: 'Refrigerator',
            description: 'Closed refrigerator',
            type: 'object',
            confidence: 0.91,
            x: 400,
            y: 200
          },
          {
            id: 'milk_1',
            label: 'Milk',
            description: 'Milk carton on counter',
            type: 'object',
            confidence: 0.93,
            x: 200,
            y: 350
          },
          {
            id: 'counter_1',
            label: 'Counter',
            description: 'Kitchen counter with various items',
            type: 'object',
            confidence: 0.87,
            x: 200,
            y: 300
          },
          {
            id: 'glass_1',
            label: 'Glass',
            description: 'Empty drinking glass',
            type: 'object',
            confidence: 0.85,
            x: 180,
            y: 320
          }
        ],
        edges: [
          {
            id: 'edge_1',
            source: 'person_1',
            target: 'counter_1',
            label: 'standing at',
            type: 'current',
            confidence: 0.9
          },
          {
            id: 'edge_2',
            source: 'milk_1',
            target: 'counter_1',
            label: 'placed on',
            type: 'current',
            confidence: 0.88
          },
          {
            id: 'edge_3',
            source: 'glass_1',
            target: 'counter_1',
            label: 'on',
            type: 'current',
            confidence: 0.85
          },
          {
            id: 'edge_4',
            source: 'person_1',
            target: 'milk_1',
            label: 'will pour into glass',
            type: 'predicted',
            confidence: 0.82,
            description: 'Person will likely pour milk into the glass'
          }
        ]
      }
    ];
  }

  // Set video chunks (would be called after processing)
  setVideoChunks(chunks: VideoChunk[]) {
    this.videoChunksSubject.next(chunks);
  }

  // Get current video chunks
  getVideoChunks(): VideoChunk[] {
    return this.videoChunksSubject.value;
  }

  // Set current chunk
  setCurrentChunk(chunk: VideoChunk | null) {
    this.currentChunkSubject.next(chunk);
  }

  // Get current chunk
  getCurrentChunk(): VideoChunk | null {
    return this.currentChunkSubject.value;
  }

  // Simulate video processing (would connect to your Python backend)
  async processVideo(videoFile: File): Promise<VideoChunk[]> {
    // This would send the video to your backend for processing
    // For now, return mock data
    return new Promise((resolve) => {
      setTimeout(() => {
        const chunks = this.getMockVideoChunks();
        this.setVideoChunks(chunks);
        resolve(chunks);
      }, 2000);
    });
  }

  // Get predictions for next chunk
  getPredictions(context: PredictionContext): Observable<any> {
    // This would call your prediction API
    // For now, return mock predictions
    return new Observable(observer => {
      setTimeout(() => {
        observer.next({
          predictions: [
            { event: 'pour_milk', probability: 0.82, description: 'Person will pour milk into glass' },
            { event: 'drink_milk', probability: 0.65, description: 'Person will drink the milk' },
            { event: 'put_away_milk', probability: 0.45, description: 'Person will put milk back in refrigerator' }
          ]
        });
        observer.complete();
      }, 500);
    });
  }

  // Update graph data for a specific chunk
  updateChunkGraph(chunkId: string, nodes: GraphNode[], edges: GraphEdge[]) {
    const chunks = this.getVideoChunks();
    const chunkIndex = chunks.findIndex(chunk => chunk.id === chunkId);
    
    if (chunkIndex !== -1) {
      chunks[chunkIndex].nodes = nodes;
      chunks[chunkIndex].edges = edges;
      this.setVideoChunks([...chunks]);
    }
  }
}
