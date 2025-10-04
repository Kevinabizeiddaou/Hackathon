# Video-to-Graph Representation System - Frontend

This Angular application provides a user interface for the video-to-graph representation agentic system. It allows users to upload videos, visualize the generated graph representations, and navigate through video chunks using an interactive slider.

## Features

- **Video Upload**: Upload video files for processing
- **Interactive Timeline**: Slider to navigate through video chunks
- **Dynamic Graph Visualization**: Real-time graph updates using D3.js
- **Node Information**: Detailed tooltips showing object descriptions and confidence scores
- **Edge Types**: Visual distinction between current relationships (solid lines) and predicted actions (dotted lines)
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

### Components
- `AppComponent`: Main application component with video upload and chunk navigation
- `GraphVisualizationComponent`: D3.js-powered graph visualization with interactive nodes and edges

### Services
- `VideoGraphService`: Handles video chunk data, graph updates, and backend communication

### Models
- `GraphNode`: Represents objects detected in video frames
- `GraphEdge`: Represents relationships and predicted actions
- `VideoChunk`: Contains graph state for a specific time segment
- `GraphData`: Combined nodes and edges for visualization

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn
- Angular CLI (optional, for development)

### Installation

1. Navigate to the frontend2 directory:
   ```bash
   cd frontend2
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser and navigate to `http://localhost:4200`

### Development Commands

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm run watch` - Build and watch for changes
- `npm test` - Run unit tests

## Integration with Backend

The frontend is designed to integrate with your Python backend system. Key integration points:

1. **Video Processing**: `VideoGraphService.processVideo()` should send video files to your chunking and analysis pipeline
2. **Graph Data**: The service expects video chunks with nodes (detected objects) and edges (relationships/predictions)
3. **Real-time Updates**: The graph visualization updates automatically when chunk data changes

### Expected Data Format

```typescript
interface VideoChunk {
  id: string;
  startTime: number;
  endTime: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  summary: string;
}
```

## Graph Visualization

The graph uses D3.js force simulation with:
- **Nodes**: Colored by type (person, object, location)
- **Edges**: Solid lines for current relationships, dotted for predictions
- **Interactions**: Drag nodes, zoom/pan, hover for details
- **Responsive**: Adapts to different screen sizes

## Mock Data

The application includes mock data for development and demonstration:
- 3 video chunks showing a kitchen scene
- Person interacting with refrigerator and objects
- Predicted actions based on context

## Customization

### Styling
- Global styles in `src/styles.scss`
- Component-specific styles in respective `.scss` files
- CSS variables for easy theme customization

### Graph Appearance
- Node colors and sizes in `GraphVisualizationComponent`
- Force simulation parameters for layout behavior
- Edge styling for different relationship types

## Next Steps

1. Connect to your Python backend APIs
2. Add video playback functionality
3. Implement real-time processing updates
4. Add export/save functionality for graph data
5. Enhance mobile responsiveness

## Technical Stack

- **Angular 17**: Modern Angular with standalone components
- **D3.js**: Data visualization and graph rendering
- **TypeScript**: Type-safe development
- **SCSS**: Enhanced CSS with variables and mixins
- **RxJS**: Reactive programming for data flow
