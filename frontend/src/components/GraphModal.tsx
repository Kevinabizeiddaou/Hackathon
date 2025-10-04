import React, { useEffect, useRef, useState } from 'react';
import { X, ChevronLeft, ChevronRight, FileText, Network } from 'lucide-react';
import cytoscape from 'cytoscape';
import { useAppStore } from '../store/useAppStore';
import { toGraph } from '../utils/graphUtils';

export const GraphModal: React.FC = () => {
  const cyRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<'graph' | 'triples' | 'objects'>('graph');
  const { 
    isModalOpen, 
    setIsModalOpen, 
    selectedChunkId, 
    chunks, 
    analysisByChunkId,
    setSelectedChunkId 
  } = useAppStore();

  const currentAnalysis = selectedChunkId ? analysisByChunkId[selectedChunkId] : null;
  const currentChunkIndex = chunks.findIndex(chunk => chunk.id === selectedChunkId);

  useEffect(() => {
    if (!isModalOpen || !currentAnalysis || !cyRef.current) return;

    const graphData = toGraph(currentAnalysis);
    
    // Clear previous graph
    cyRef.current.innerHTML = '';

    const cy = cytoscape({
      container: cyRef.current,
      elements: [
        ...graphData.nodes.map(node => ({
          data: { id: node.id, label: node.label, desc: node.desc }
        })),
        ...graphData.edges.map(edge => ({
          data: { 
            id: edge.id, 
            source: edge.source, 
            target: edge.target, 
            label: edge.label 
          }
        }))
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#3b82f6',
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'color': 'white',
            'font-size': '12px',
            'font-weight': 'bold',
            'width': '60px',
            'height': '60px',
            'border-width': '2px',
            'border-color': '#1e40af',
            'text-outline-width': '2px',
            'text-outline-color': '#1e40af'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': '3px',
            'line-color': '#6b7280',
            'target-arrow-color': '#6b7280',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': '-10px',
            'color': '#374151'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'background-color': '#ef4444',
            'border-color': '#dc2626',
            'border-width': '3px'
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: true,
        animationDuration: 1000,
        fit: true,
        padding: 20
      }
    });

    // Add hover effects
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      node.style('background-color', '#ef4444');
    });

    cy.on('mouseout', 'node', (evt) => {
      const node = evt.target;
      node.style('background-color', '#3b82f6');
    });

    return () => {
      cy.destroy();
    };
  }, [isModalOpen, currentAnalysis]);

  const handlePrevious = () => {
    if (currentChunkIndex > 0) {
      setSelectedChunkId(chunks[currentChunkIndex - 1].id);
    }
  };

  const handleNext = () => {
    if (currentChunkIndex < chunks.length - 1) {
      setSelectedChunkId(chunks[currentChunkIndex + 1].id);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') handlePrevious();
    if (e.key === 'ArrowRight') handleNext();
    if (e.key === 'Escape') setIsModalOpen(false);
  };

  if (!isModalOpen || !currentAnalysis) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && setIsModalOpen(false)}
      onKeyDown={handleKeyDown}
      tabIndex={-1}
    >
      <div className="bg-white rounded-3xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-bold text-gray-800">Scene Analysis</h2>
            <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
              Chunk #{selectedChunkId?.slice(-3)}
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              disabled={currentChunkIndex <= 0}
              className="p-2 rounded-full hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <button
              onClick={handleNext}
              disabled={currentChunkIndex >= chunks.length - 1}
              className="p-2 rounded-full hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsModalOpen(false)}
              className="p-2 rounded-full hover:bg-gray-100"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex h-[calc(90vh-120px)]">
          {/* Graph Panel */}
          <div className="flex-1 p-6">
            <div className="h-full bg-gray-50 rounded-2xl border border-gray-200">
              <div ref={cyRef} className="w-full h-full rounded-2xl" />
            </div>
          </div>

          {/* Details Panel */}
          <div className="w-96 border-l border-gray-200 p-6 overflow-y-auto">
            {/* Tabs */}
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setActiveTab('graph')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'graph' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Network className="w-4 h-4" />
                Graph
              </button>
              <button
                onClick={() => setActiveTab('triples')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === 'triples' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <FileText className="w-4 h-4" />
                Relations
              </button>
            </div>

            {/* Narrative */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">What's Happening</h3>
              <p className="text-gray-700 bg-blue-50 p-4 rounded-xl">
                {currentAnalysis.narrative}
              </p>
            </div>

            {/* Relations Table */}
            {activeTab === 'triples' && (
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Relations</h3>
                <div className="space-y-2">
                  {currentAnalysis.relations.map((relation, index) => (
                    <div key={index} className="bg-gray-50 p-3 rounded-lg">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="font-medium text-blue-600">{relation[0]}</span>
                        <span className="text-gray-500">→</span>
                        <span className="font-medium text-green-600">{relation[1]}</span>
                        <span className="text-gray-500">→</span>
                        <span className="font-medium text-purple-600">{relation[2]}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Objects List */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Objects</h3>
              <div className="space-y-3">
                {currentAnalysis.objects.map((obj, index) => (
                  <div key={index} className="bg-white border border-gray-200 p-3 rounded-lg">
                    <div className="font-medium text-gray-800">{obj.name}</div>
                    {obj.description && (
                      <div className="text-sm text-gray-600 mt-1">{obj.description}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
