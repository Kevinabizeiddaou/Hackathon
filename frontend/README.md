# Video Scene Graph Explorer

A React-based web application that analyzes video content and visualizes scene relationships as interactive graphs.

## Features

- **Video Upload**: Upload MP4, MOV, or AVI video files
- **Chunk Generation**: Automatically splits videos into analyzable segments
- **Scene Analysis**: Extracts objects and their relationships from video chunks
- **Interactive Graph Visualization**: Uses Cytoscape.js to display scene graphs
- **Mock Mode**: Demo mode with sample data for testing without backend
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Built with TailwindCSS and smooth animations

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: TailwindCSS with custom components
- **State Management**: Zustand
- **Graph Visualization**: Cytoscape.js
- **Icons**: Lucide React
- **Backend Integration**: FastAPI (Python)

## Project Structure

```
src/
├── components/          # React components
│   ├── UploadPanel.tsx  # Video upload interface
│   ├── ChunkTimeline.tsx # Chunk selection timeline
│   └── GraphModal.tsx   # Graph visualization modal
├── store/               # State management
│   └── useAppStore.ts   # Zustand store
├── services/            # API integration
│   └── api.ts          # API service layer
├── utils/               # Utility functions
│   └── graphUtils.ts   # Graph data transformation
├── data/               # Mock data
│   └── mockData.ts     # Sample chunks and analysis
├── types.ts            # TypeScript type definitions
└── App.tsx             # Main application component
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Start Development Server**:
   ```bash
   npm run dev
   ```

3. **Open Browser**: Navigate to `http://localhost:5173`

## Usage

1. **Upload Video**: Click the upload area and select a video file
2. **Generate Chunks**: Click "Generate Chunks" to split the video
3. **Select Chunk**: Click on any chunk card to analyze it
4. **View Graph**: The modal opens showing:
   - Interactive graph visualization
   - Narrative description
   - Relations table
   - Objects list
5. **Navigate**: Use arrow keys or buttons to switch between chunks

## Mock Mode

The application starts in mock mode with sample data. Toggle the "Mock Mode" switch to:
- **ON**: Use sample data (no backend required)
- **OFF**: Connect to FastAPI backend at `http://localhost:8000`

## Backend Integration

The app integrates with a FastAPI backend that provides:
- `POST /chunk`: Video chunking endpoint
- `POST /analyze`: Scene analysis endpoint

See the main `main.py` file for the backend implementation.

## Key Components

### UploadPanel
- File picker with drag-and-drop styling
- Mock/Live mode toggle
- Generate chunks functionality

### ChunkTimeline
- Responsive grid of chunk cards
- Thumbnail placeholders
- Time range display
- Click to analyze

### GraphModal
- Full-screen modal with graph visualization
- Tabbed interface (Graph/Relations)
- Navigation between chunks
- Keyboard shortcuts (←/→ arrows, Escape)

## Styling

Uses TailwindCSS with custom component classes:
- `.glass-card`: Glassmorphism effect
- `.btn-primary` / `.btn-secondary`: Button styles
- `.chunk-card`: Chunk card styling with hover effects

## Development

- **TypeScript**: Full type safety
- **ESLint**: Code linting
- **Hot Reload**: Instant updates during development
- **Responsive**: Mobile-first design approach

## Demo Data

The mock mode includes sample data for:
- 4 video chunks (2-second segments)
- Dance scene analysis with person-to-person relationships
- Various objects (people, cars, dogs)
- Spatial and interaction relations

## Future Enhancements

- Real video thumbnail generation
- Timeline scrubber
- Node pinning functionality
- Graph export (PNG/SVG)
- Dark mode toggle
- Advanced graph layouts
- Real-time analysis progress
