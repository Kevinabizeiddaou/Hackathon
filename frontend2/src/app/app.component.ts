import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GraphVisualizationComponent } from './components/graph-visualization/graph-visualization.component';
import { VideoChunk, GraphData } from './models/graph.models';
import { VideoGraphService } from './services/video-graph.service';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, GraphVisualizationComponent, FormsModule],
  template: `
    <div class="app-container">
      <header class="app-header">
        <h1>Video-to-Graph Representation System</h1>
        <p>Analyze video content through dynamic graph visualization</p>
      </header>

      <main class="main-content">
        <!-- Video Upload Section -->
        <section class="upload-section card">
          <h2>Upload Video</h2>
          <div class="upload-area" [class.has-file]="selectedFile">
            <input #fileInput type="file" accept="video/*" 
                   (change)="onFileSelected($event)" hidden>
            
            <!-- Video Preview (shown when file is selected) -->
            <div class="video-preview" *ngIf="selectedFile && videoUrl">
              <video #videoPlayer 
                     [src]="videoUrl" 
                     [currentTime]="currentChunk?.startTime || 0"
                     controls
                     muted>
              </video>
              <div class="video-info">
                <p class="file-name">{{ selectedFile.name }}</p>
                <p class="chunk-info" *ngIf="currentChunk">
                  Current: Chunk {{ currentChunkIndex + 1 }} ({{ formatTime(currentChunk.startTime) }} - {{ formatTime(currentChunk.endTime) }})
                </p>
              </div>
            </div>
            
            <!-- Upload Placeholder (shown when no file selected) -->
            <div class="upload-content" *ngIf="!selectedFile" (click)="fileInput.click()">
              <div class="upload-icon">📹</div>
              <p>Click to upload video file</p>
            </div>
          </div>
          
          <div class="upload-actions">
            <button class="btn btn-secondary" 
                    *ngIf="selectedFile"
                    (click)="clearVideo()">
              Change Video
            </button>
            <button class="btn btn-primary" 
                    [disabled]="!selectedFile || isProcessing"
                    (click)="processVideo()">
              {{ isProcessing ? 'Processing...' : 'Process Video' }}
            </button>
          </div>
        </section>

        <!-- Video Chunk Slider Section -->
        <section class="slider-section card" *ngIf="videoChunks.length > 0">
          <h2>Video Timeline</h2>
          <div class="slider-container">
            <div class="slider-info">
              <span>Chunk {{ currentChunkIndex + 1 }} of {{ videoChunks.length }}</span>
              <span>{{ formatTime(currentChunk?.startTime) }} - {{ formatTime(currentChunk?.endTime) }}</span>
            </div>
            <input type="range" 
                   class="slider"
                   [min]="0" 
                   [max]="videoChunks.length - 1"
                   [(ngModel)]="currentChunkIndex"
                   (input)="onChunkChange($event)">
            <div class="chunk-summary" *ngIf="currentChunk">
              <h3>Chunk Summary</h3>
              <p>{{ currentChunk.summary }}</p>
            </div>
          </div>
        </section>

        <!-- Graph Visualization Section -->
        <section class="graph-section card" *ngIf="currentGraphData">
          <h2>Graph Visualization</h2>
          <div class="graph-display">
            <app-graph-visualization 
              [graphData]="currentGraphData"
              [width]="600"
              [height]="450">
            </app-graph-visualization>
          </div>
        
          <!-- Legend -->
          <div class="legend">
            <h3>Legend</h3>
            <div class="legend-items">
              <div class="legend-item">
                <div class="legend-line solid"></div>
                <span>Current Relationships</span>
              </div>
              <div class="legend-item">
                <div class="legend-line dotted"></div>
                <span>Predicted Actions</span>
              </div>
            </div>
          </div>
        </section>

        
        <!-- Processing Status -->
        <section class="status-section" *ngIf="isProcessing">
          <div class="processing-indicator">
            <div class="spinner"></div>
            <p>Processing video chunks and generating graph representations...</p>
          </div>
        </section>
      </main>
    </div>
  `,
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;
  
  selectedFile: File | null = null;
  isProcessing = false;
  videoChunks: VideoChunk[] = [];
  currentChunkIndex = 0;
  currentGraphData: GraphData | null = null;
  videoUrl: string | null = null;

  constructor(private videoGraphService: VideoGraphService) {}

  ngOnInit() {
    // Initialize with mock data for development
    this.loadMockData();
  }

  get currentChunk(): VideoChunk | null {
    return this.videoChunks[this.currentChunkIndex] || null;
  }

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      this.selectedFile = file;
      // Create video URL for preview
      this.videoUrl = URL.createObjectURL(file);
    }
  }

  clearVideo() {
    this.selectedFile = null;
    if (this.videoUrl) {
      URL.revokeObjectURL(this.videoUrl);
      this.videoUrl = null;
    }
    this.videoChunks = [];
    this.currentChunkIndex = 0;
    this.currentGraphData = null;
  }

  async processVideo() {
    if (!this.selectedFile) return;

    this.isProcessing = true;
    try {
      // In a real implementation, this would send the video to your backend
      // For now, we'll simulate processing with mock data
      await this.simulateVideoProcessing();
    } catch (error) {
      console.error('Error processing video:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  onChunkChange(event: any) {
    this.currentChunkIndex = parseInt(event.target.value);
    this.updateGraphData();
    this.updateVideoPreview();
  }

  private updateGraphData() {
    if (this.currentChunk) {
      this.currentGraphData = {
        nodes: this.currentChunk.nodes,
        edges: this.currentChunk.edges
      };
    }
  }

  private updateVideoPreview() {
    if (this.videoPlayer && this.currentChunk && this.videoUrl) {
      // Set video time to the start of the current chunk
      setTimeout(() => {
        const video = this.videoPlayer.nativeElement;
        video.currentTime = this.currentChunk!.startTime;
      }, 100);
    }
  }

  private async simulateVideoProcessing() {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    this.loadMockData();
  }

  private loadMockData() {
    this.videoChunks = this.videoGraphService.getMockVideoChunks();
    if (this.videoChunks.length > 0) {
      this.currentChunkIndex = 0;
      this.updateGraphData();
    }
  }

  formatTime(seconds: number | undefined): string {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
