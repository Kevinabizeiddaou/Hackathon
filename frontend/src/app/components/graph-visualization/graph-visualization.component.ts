import { Component, Input, OnChanges, SimpleChanges, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as d3 from 'd3';
import { GraphData, GraphNode, GraphEdge } from '../../models/graph.models';

@Component({
  selector: 'app-graph-visualization',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="graph-container">
      <svg #svgElement [attr.width]="width" [attr.height]="height"></svg>
      <div class="node-tooltip" #tooltip></div>
    </div>
  `,
  styleUrls: ['./graph-visualization.component.scss']
})
export class GraphVisualizationComponent implements OnChanges, AfterViewInit {
  @Input() graphData: GraphData | null = null;
  @Input() width = 800;
  @Input() height = 600;

  @ViewChild('svgElement', { static: true }) svgElement!: ElementRef<SVGElement>;
  @ViewChild('tooltip', { static: true }) tooltip!: ElementRef<HTMLDivElement>;

  private svg: any;
  private simulation: any;

  ngAfterViewInit() {
    this.initializeSvg();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['graphData'] && this.svg) {
      this.updateVisualization();
    }
  }

  private initializeSvg() {
    this.svg = d3.select(this.svgElement.nativeElement);
    
    // Create main group for graph elements (no interaction)
    this.svg.append('g').attr('class', 'graph-group');

    if (this.graphData) {
      this.updateVisualization();
    }
  }

  private updateVisualization() {
    if (!this.graphData || !this.svg) return;

    const graphGroup = this.svg.select('.graph-group');
    
    // Clear previous visualization
    graphGroup.selectAll('*').remove();

    // Set up simple static positions for nodes
    this.setupNodePositions();

    // Create links (edges) first so they appear behind nodes
    const links = graphGroup.selectAll('.link')
      .data(this.graphData.edges)
      .enter().append('g')
      .attr('class', 'link-group');

    // Add edge lines
    links.append('line')
      .attr('class', (d: GraphEdge) => `link ${d.type}`)
      .attr('stroke', (d: GraphEdge) => d.type === 'predicted' ? '#ed8936' : '#4299e1')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', (d: GraphEdge) => d.type === 'predicted' ? '5,5' : '0')
      .attr('x1', (d: GraphEdge) => this.getNodeById(d.source)?.x || 0)
      .attr('y1', (d: GraphEdge) => this.getNodeById(d.source)?.y || 0)
      .attr('x2', (d: GraphEdge) => this.getNodeById(d.target)?.x || 0)
      .attr('y2', (d: GraphEdge) => this.getNodeById(d.target)?.y || 0);

    // Add edge labels
    links.append('text')
      .attr('class', 'link-label')
      .attr('text-anchor', 'middle')
      .attr('dy', -5)
      .attr('x', (d: GraphEdge) => {
        const source = this.getNodeById(d.source);
        const target = this.getNodeById(d.target);
        return ((source?.x || 0) + (target?.x || 0)) / 2;
      })
      .attr('y', (d: GraphEdge) => {
        const source = this.getNodeById(d.source);
        const target = this.getNodeById(d.target);
        return ((source?.y || 0) + (target?.y || 0)) / 2;
      })
      .text((d: GraphEdge) => d.label)
      .style('font-size', '12px')
      .style('fill', '#4a5568')
      .style('font-weight', '500');

    // Create nodes (static, no dragging)
    const nodes = graphGroup.selectAll('.node')
      .data(this.graphData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', (d: GraphNode) => `translate(${d.x},${d.y})`);

    // Add node circles (larger for better text fit)
    nodes.append('circle')
      .attr('r', 25)
      .attr('fill', (d: GraphNode) => this.getNodeColor(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 3)
      .style('cursor', 'pointer');

    // Add node labels (better text fitting)
    nodes.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .text((d: GraphNode) => this.formatNodeLabel(d.label))
      .style('font-size', '10px')
      .style('font-weight', '600')
      .style('fill', 'white')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 1px 2px rgba(0,0,0,0.5)');

    // Add tooltips
    nodes
      .on('mouseover', (event: MouseEvent, d: GraphNode) => this.showTooltip(event, d))
      .on('mouseout', () => this.hideTooltip());
  }

  private setupNodePositions() {
    if (!this.graphData) return;
    
    const nodeCount = this.graphData.nodes.length;
    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const minDistance = 100; // Minimum distance between nodes
    const nodeRadius = 25; // Node visual radius
    
    if (nodeCount === 1) {
      // Single node in center
      this.graphData.nodes[0].x = centerX;
      this.graphData.nodes[0].y = centerY;
    } else if (nodeCount === 2) {
      // Two nodes side by side
      this.graphData.nodes[0].x = centerX - minDistance / 2;
      this.graphData.nodes[0].y = centerY;
      this.graphData.nodes[1].x = centerX + minDistance / 2;
      this.graphData.nodes[1].y = centerY;
    } else {
      // Circular layout with minimum distance
      const radius = Math.max(
        minDistance * nodeCount / (2 * Math.PI), 
        Math.min(this.width, this.height) / 3
      );
      
      this.graphData.nodes.forEach((node, index) => {
        const angle = (index / nodeCount) * 2 * Math.PI;
        node.x = centerX + radius * Math.cos(angle);
        node.y = centerY + radius * Math.sin(angle);
      
        // Ensure nodes stay within canvas bounds
        node.x = Math.max(nodeRadius + 10, Math.min(this.width - nodeRadius - 10, node.x));
        node.y = Math.max(nodeRadius + 10, Math.min(this.height - nodeRadius - 10, node.y));
      });      
    }
  }

  private getNodeById(id: string): GraphNode | undefined {
    return this.graphData?.nodes.find(node => node.id === id);
  }

  private formatNodeLabel(label: string): string {
    // Ensure label fits well within the circle
    if (label.length <= 6) {
      return label;
    } else if (label.length <= 9) {
      return label.substring(0, 6) + '...';
    } else {
      return label.substring(0, 5) + '...';
    }
  }

  private getNodeColor(type: string): string {
    const colors = {
      'object': '#4299e1',
      'person': '#48bb78',
      'location': '#ed8936',
      'other': '#9f7aea'
    };
    return colors[type as keyof typeof colors] || colors.other;
  }

  private showTooltip(event: any, node: GraphNode) {
    const tooltip = d3.select(this.tooltip.nativeElement);
    
    tooltip.style('opacity', 1)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px')
      .html(`
        <div class="tooltip-content">
          <h4>${node.label}</h4>
          <p><strong>Type:</strong> ${node.type}</p>
          <p><strong>Confidence:</strong> ${(node.confidence * 100).toFixed(1)}%</p>
          <p><strong>Description:</strong> ${node.description}</p>
        </div>
      `);
  }

  private hideTooltip() {
    d3.select(this.tooltip.nativeElement).style('opacity', 0);
  }

}
