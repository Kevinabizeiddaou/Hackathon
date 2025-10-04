# Overview

Current video captioning systems are constrained by domain-labeled datasets (e.g., crime-only corpora), yielding agents that simply mirror their training scope rather than understanding open-world scenes.
Our system delivers **domain-agnostic video annotation** by coupling **semantic segmentation** with an agentic **orchestrator/mapper**, enabling robust label grounding across diverse footage.
Beyond naming categories, we provide **instance-level disambiguation** (e.g., **person 1**, **person 2**, **person 3**) so identical classes within the same frame remain uniquely trackable.
For each video chunk, we construct a **graph of actions** where nodes are **labels** and edges are **observed interactions**, producing an interpretable, queryable structure.
In the **final chunk**, the graph additionally renders **predicted interactions** as **dotted edges**, turning passive description into forward-looking situational awareness.
These predictions integrate **k‑means–based recommendation** over agent states with **temporal context** from prior chunks—akin to sequence models such as **RNNs**—to surface the **top‑three next actions**.
The result is a **general-purpose, instance-aware, predictive video understanding pipeline** that transcends dataset silos and supports downstream analytics and decision-making.

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

Chunking Agent	/ Splits video into time-based states

Object Detector (HF SegFormer) /	Extracts and segments visual objects

Describer Agent	/ Provides semantic labels and contextual understanding

Action Describer Agent / Infers what is happening between objects

Predictive Agent / Forecasts the next possible interactions or events


![WhatsApp Image 2025-10-04 at 10 40 00 PM](https://github.com/user-attachments/assets/0470c318-cdc2-434d-970b-b7af3d8d6d3b)



## Sample Input/Output

### Input: https://www.youtube.com/shorts/3R7_dTzq2mE 
### Output:
```
[main] Chunking video…
[main] Created 3 chunks from: C:\Users\Kareem Hassani\OneDrive\Desktop\College\4 - Year Four\Fall\EECE 503P\FINAL\Hackathon\backend\stop_horsin_around.mp4
[main] Loading SegFormer…
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

[main] === Processing chunk 0 ===
[main] agent objects: 6
[main] HF segments: 20 @ (608x1080)
[main] mapping unified_objects: 20
[main] interactions: 0
[front] chunk_1 FinalLabels: ['airplane', 'bottle', 'cabinet', 'conveyer belt', 'fence', 'floor', 'fountain', 'person', 'shelf', 'sink', 'tray', 'wall']
[front] chunk_1 Interactions (0):

[main] === Processing chunk 1 ===
[main] agent objects: 3
[main] HF segments: 26 @ (608x1080)
[main] mapping unified_objects: 3
[main] interactions: 2
[front] chunk_2 FinalLabels: ['hand', 'horse', 'watermelon']
[front] chunk_2 Interactions (2):
         - horse ↔ watermelon :: horse is eating the watermelon
         - hand ↔ watermelon :: hand is holding the watermelon

[main] === Processing chunk 2 ===
[main] agent objects: 3
[main] HF segments: 32 @ (608x1080)
[main] mapping unified_objects: 3
[main] interactions: 1
[front] chunk_3 FinalLabels: ['door', 'horse', 'watermelon']
[front] chunk_3 Interactions (1):
         - horse ↔ watermelon :: horse is eating the watermelon

[main] === PIPELINE RESULT (keys only) ===
{
  "video_path": "C:\\Users\\Kareem Hassani\\OneDrive\\Desktop\\College\\4 - Year Four\\Fall\\EECE 503P\\FINAL\\Hackathon\\backend\\stop_horsin_around.mp4",
  "num_chunks": 3,
  "per_chunk_keys": [
    [
      "chunk_index",
      "agent_json",
      "agent_raw",
      "hf_segments",
      "image_size",
      "mapping",
      "interactions"
    ],
    [
      "chunk_index",
      "agent_json",
      "agent_raw",
      "hf_segments",
      "image_size",
      "mapping",
      "interactions"
    ],
    [
      "chunk_index",
      "agent_json",
      "agent_raw",
      "hf_segments",
      "image_size",
      "mapping",
      "interactions"
    ]
  ],
  "chunk_graph_keys": [
    "chunk_1",
    "chunk_2",
    "chunk_3"
  ]
}
['chunk_1', 'chunk_2', 'chunk_3']
FinalLabels: ['hand', 'horse', 'watermelon']
Interactions:
  (horse, watermelon, horse is eating the watermelon)
  (hand, watermelon, hand is holding the watermelon)

[predict] Inputs for predict_next_events()
[predict] previous_summaries:
  - No interactions
  - horse eating watermelon; hand holding watermelon
[predict] current_summary: horse eating watermelon
[predict] current_objects: ['door', 'horse', 'watermelon']
[predict] NextEvents: ['horse continues eating watermelon', 'horse moves closer to the door', 'horse finishes eating watermelon']
```

<img width="1200" height="900" alt="chunk_1" src="https://github.com/user-attachments/assets/856c05d8-d5ec-4509-818c-d0014408cf73" />

<img width="1200" height="900" alt="chunk_2" src="https://github.com/user-attachments/assets/c690b5d5-cd56-4925-ba4d-c362c56384d7" />

<img width="1200" height="900" alt="chunk_3" src="https://github.com/user-attachments/assets/872eb7fd-d534-49e5-8601-ed293d2dadcb" />


If you wish to try it for yourself, you can run main.py (with any video, just change the video path in main) and observe the graphs that are generated in the same directory.
