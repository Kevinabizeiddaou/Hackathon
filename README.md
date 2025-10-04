# Overview
This project transforms raw videos into structured graph representations that describe the evolving relationships and actions happening on screen.
Each frame sequence (chunk) becomes a graph state — where:

Nodes represent detected objects,

Edges represent interactions or actions between them, and

The graph evolves over time as scenes change.

The system not only interprets what’s happening in the video, but also predicts the next most probable events, allowing for dynamic scene understanding, reasoning, and forecasting.

## How It Works
1. Video Chunking

The pipeline begins by dividing the input video into temporal chunks.
Each chunk captures a consistent scene or motion window.

[main] Chunking video…
[main] Created 3 chunks from: stop_horsin_around.mp4


Each chunk acts as a snapshot of the world, forming the basis for graph state generation.

2. Object Detection (Node Creation)

A SegFormer (Hugging Face segmentation model) processes each chunk to identify visual entities in the frame.

Example output:

[front] chunk_1 FinalLabels:
['airplane', 'bottle', 'cabinet', 'fountain', 'person', 'shelf', ...]


Each detected object becomes a graph node, enriched with contextual descriptions provided by an internal describer agent.

3. Interaction Detection (Edge Creation)

A pair of specialized agents work together to determine relationships:

Proximity Estimator: Evaluates spatial and contextual closeness between objects.

Action Describer Agent: Interprets what’s happening — generating relational statements like:

“horse is eating the watermelon”

“hand is holding the watermelon”

Example output:

[front] chunk_2 Interactions (2):
  - horse ↔ watermelon :: horse is eating the watermelon
  - hand ↔ watermelon :: hand is holding the watermelon


Edges in the graph correspond to these detected actions.

4. Graph State Evolution

As the system processes each chunk, the graph updates in real time:

Nodes appear or disappear as objects enter or leave the frame.

Edges are created, removed, or updated based on ongoing actions.

Example transition:

chunk_2 → chunk_3
Removed: hand
New: door
Ongoing: horse ↔ watermelon (eating)

5. Next-Action Prediction

After processing the current chunk, the system predicts what might happen next.

It uses:

The current graph structure (nodes + edges)

Object proximity

Detected actions and their history

Example:

[predict] NextEvents:
  - horse continues eating watermelon
  - horse moves closer to the door
  - horse finishes eating watermelon


This predictive ability enables anticipation of future scene dynamics.

## System Architecture
Agent / Module	Role

Chunking Agent	Splits video into time-based states

Object Detector (HF SegFormer)	Extracts and segments visual objects

Describer Agent	Provides semantic labels and contextual understanding

Action Describer Agent	Infers what is happening between objects

Predictive Agent	Forecasts the next possible interactions or events



