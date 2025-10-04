# ===================== main.py (explicit, line-by-line) ======================
# Video → Chunks → (Agent JSON, HF JSON) → Mapping JSON → Interaction JSON
# + Packs per-chunk "FinalLabels" and "Interactions" for graph building.
# ============================================================================
import os, sys, json
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from cvclip import CVClip

# Make sure Python can import your helper modules (adjust if needed)
sys.path.insert(0, "/mnt/data")

# --- Import helpers ----------------------------------------------------------
from chunker import chunk_video, close_video_chunks  # motion-based splitter
from agent_segmentation import (                     # LLM describer (objects/relations)
    describe_chunks_on_screen,
    ON_SCREEN_ONLY_PROMPT,
    build_agent_call,
)
from hf_segmentation import (                        # HF SegFormer (single chunk)
    load_segformer,
    segments_from_chunk,
)
from orchestra_hf_agent_segmentation import (        # LLM mapping (agent ↔ HF)
    map_objects_via_llm_v2,
)
from agent_spec_captioning import (                  # Interactions JSON (single chunk)
    set_openai_client,
    describe_interactions_for_labels,
)

# --------------------------- Configuration -----------------------------------
# ⚠️ If hardcoding, do as below; otherwise prefer env var OPENAI_API_KEY
client = OpenAI(api_key="...")
set_openai_client(client)  # for the interaction module (required once)

# 2) Input video
video_path = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\4 - Year Four\Fall\EECE 503P\Hackathon\Hackathon\backend\stop_horsin_around.mp4"  # <-- change this

# 3) Chunking params
min_chunk         = 4
max_chunk         = 24
motion_threshold  = 0.0375
frame_sample_rate = 40

# 4) Agent (describer) params
describer_model   = "gpt-4o"
frames_per_chunk  = 4
max_frame_width   = 640

# 5) HF segmentation params
segformer_model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
min_pixels         = 200
keep_labels        = None  # or e.g. ["person","horse","fence","tree"]

# 6) Mapping (orchestrator) params
mapping_model   = "gpt-4o"
k_neighbors     = 5

# 7) Interaction extractor params
interaction_model         = "gpt-4o"
n_frames_for_interactions = 2

# ------------------------------- Pipeline ------------------------------------
print("[main] Chunking video…")
chunks = chunk_video(
    video_path=video_path,
    min_chunk=min_chunk,
    max_chunk=max_chunk,
    motion_threshold=motion_threshold,
    frame_sample_rate=frame_sample_rate,
)
print(f"[main] Created {len(chunks)} chunks from: {video_path}")

# Build JSON-enforced vision describer call (LLM)
agent_call = build_agent_call(client, model=describer_model)

# Load SegFormer once
print("[main] Loading SegFormer…")
seg_processor, seg_model = load_segformer(segformer_model_id)

# Storage for step-by-step artifacts (easy to inspect while debugging)
agent_descriptions: List[Dict[str, Any]] = []
hf_segments_by_chunk: List[List[Dict[str, Any]]] = []
image_sizes: List[Tuple[int, int]] = []
mappings: List[Dict[str, Any]] = []
interactions_by_chunk: List[Dict[str, Any]] = []

# 👉 NEW: Per-chunk graph payload the frontend expects
# Format: {"chunk_1": {"FinalLabels":[...], "Interactions":[(l1,l2,desc), ...]}, ...}
ChunkGraphs: Dict[str, Dict[str, Any]] = {}

try:
    for idx, clip in enumerate(chunks):
        print(f"\n[main] === Processing chunk {idx} ===")

        # (1) Agent describer for THIS chunk
        desc_list = describe_chunks_on_screen(
            video_chunks=[clip],  # pass a list with ONE chunk
            agent_call=agent_call,
            prompt=ON_SCREEN_ONLY_PROMPT,
            frames_per_chunk=frames_per_chunk,
            max_width=max_frame_width,
        )
        agent_entry = desc_list[0]
        agent_json = agent_entry["json"]
        agent_raw = agent_entry["raw"]
        agent_descriptions.append(agent_entry)
        objects_agent = agent_json.get("objects", [])
        print(f"[main] agent objects: {len(objects_agent)}")

        # (2) HF segmentation for THIS chunk (middle frame)
        segs, (W, H) = segments_from_chunk(
            clip=clip,
            seg_processor=seg_processor,
            seg_model=seg_model,
            max_width=max_frame_width,
            min_pixels=min_pixels,
            keep_labels=keep_labels,
            rle_order="C",
        )
        hf_segments_by_chunk.append(segs)
        image_sizes.append((W, H))
        print(f"[main] HF segments: {len(segs)} @ ({W}x{H})")

        # (3) Mapping (agent ↔ HF) for THIS chunk
        mapping = map_objects_via_llm_v2(
            client=client,
            objects_agent=objects_agent,
            segments_api=segs,
            image_size=(W, H),
            k_neighbors=k_neighbors,
            model=mapping_model,
        )
        mappings.append(mapping)
        unified = mapping.get("unified_objects", [])
        print(f"[main] mapping unified_objects: {len(unified)}")

        # Allowed labels for interactions (unique label names from mapping)
        allowed_labels = sorted({
            (u.get("label") or "").lower()
            for u in unified
            if u.get("label")
        })

        # (4) Interactions (only among allowed labels) for THIS chunk
        mapping_with_meta = {**mapping, "meta": {"image_size": [W, H]}}
        interactions = describe_interactions_for_labels(
            clip=clip,
            mapping_chunk=mapping_with_meta,
            include_labels=allowed_labels,
            n_frames=n_frames_for_interactions,
            model=interaction_model,
        )
        interactions_by_chunk.append(interactions)
        inter_list = interactions.get("interactions", [])
        print(f"[main] interactions: {len(inter_list)}")

        # ---------------- NEW: Pack the frontend-friendly per-chunk payload ----------------
        # FinalLabels = all labels from orchestrator unified output
        FinalLabels = sorted({(u.get("label") or "").lower() for u in unified if u.get("label")})

        # Interactions = tuples (label1, label2, description)
        Interactions = []
        for it in inter_list:
            labs = it.get("labels", [])
            desc = it.get("description", "")
            if len(labs) >= 2:
                Interactions.append((str(labs[0]).lower(), str(labs[1]).lower(), str(desc)))

        # Save under "chunk_1", "chunk_2", ...
        chunk_key = f"chunk_{idx+1}"
        ChunkGraphs[chunk_key] = {
            "FinalLabels": FinalLabels,
            "Interactions": Interactions,
        }

        # Pretty print the essentials for this chunk
        print(f"[front] {chunk_key} FinalLabels: {FinalLabels}")
        print(f"[front] {chunk_key} Interactions ({len(Interactions)}):")
        for t in Interactions:
            print(f"         - {t[0]} ↔ {t[1]} :: {t[2]}")

finally:
    # Always free subclip handles
    close_video_chunks(chunks)

# (5) Assemble a final result dict you can save/inspect (kept from your original)
pipeline_result = {
    "video_path": video_path,
    "num_chunks": len(chunks),
    "by_chunk": [
        {
            "chunk_index": i,
            "agent_json": agent_descriptions[i]["json"],     # model (1)
            "agent_raw": agent_descriptions[i]["raw"],
            "hf_segments": hf_segments_by_chunk[i],          # model (2)
            "image_size": list(image_sizes[i]),
            "mapping": mappings[i],                          # unified (3)
            "interactions": interactions_by_chunk[i],        # final JSON
        }
        for i in range(len(chunks))
    ],
    # 👉 NEW: front-end friendly dict
    "ChunkGraphs": ChunkGraphs,
}

# Final quick look (keys only)
print("\n[main] === PIPELINE RESULT (keys only) ===")
print(json.dumps({
    "video_path": pipeline_result["video_path"],
    "num_chunks": pipeline_result["num_chunks"],
    "per_chunk_keys": [list(c.keys()) for c in pipeline_result["by_chunk"]],
    "chunk_graph_keys": list(ChunkGraphs.keys()),
}, indent=2))

# Print all chunk keys
print(list(ChunkGraphs.keys()))
# Example: ['chunk_1', 'chunk_2', 'chunk_3']

# Get chunk_2 payload
c2 = ChunkGraphs["chunk_2"]
print("FinalLabels:", c2["FinalLabels"])
print("Interactions:")
for (l1, l2, desc) in c2["Interactions"]:
    print(f"  ({l1}, {l2}, {desc})")
PREDICTION_MODEL = "gpt-4.1-mini"  
PREDICTIONS_TOP_K = 3

# ---- Next-event predictor (LLM) ----
from typing import List

def predict_next_events(previous_summaries: List[str], current_summary: str, current_objects: List[str], k: int = 3) -> List[str]:
    """
    Predicts the top k next possible events in a video scene using an LLM.
    Returns a list[str], one event per line.
    """
    prompt = f"""
You are an action prediction model for a video-to-graph system.

The video is processed into chunks. Each chunk has:
- A summary of actions
- A set of objects detected on screen

You will be given:
1. Summaries of previous chunks
2. The current chunk summary
3. The current objects in the scene

Your task is:
- Predict the most likely next events in the *next chunk*.
- If the scene is stable and nothing is likely to change, say "No significant change" or describe small, realistic variations.
- Do NOT invent objects outside the given object list.
- Keep predictions grounded in what has been happening.

---

Previous summaries:
{chr(10).join(f"- {s}" for s in previous_summaries)}

Current chunk summary:
{current_summary}

Current objects in scene:
{', '.join(current_objects)}

Now, list the most likely next events (one per line):
""".strip()

    resp = client.chat.completions.create(
        model=PREDICTION_MODEL,
        messages=[
            {"role": "system", "content": "You are a video action predictor for a graph-based video understanding system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=1,
        n=1,
    )
    text_output = resp.choices[0].message.content.strip()
    events = [line.strip("-• ").strip() for line in text_output.split("\n") if line.strip()]
    return events[:k]
def _chunk_summary_from_interactions(inter_json: Dict[str, Any]) -> str:
    """
    Make a compact summary string from the interactions JSON.
    Falls back to 'No interactions' when empty.
    """
    interactions = inter_json.get("interactions", []) if inter_json else []
    if not interactions:
        return "No interactions"
    parts = []
    for it in interactions[:8]:  # keep it short
        labs = it.get("labels") or []
        action = (it.get("action") or "").strip()
        desc = (it.get("description") or "").strip()
        if action and len(labs) >= 2:
            parts.append(f"{labs[0]} {action} {labs[1]}")
        elif desc:
            parts.append(desc)
    return "; ".join(parts) if parts else "No interactions"
# ---------------- PREDICTION STEP (last chunk only) ----------------
if len(chunks) > 0:
    last_idx = len(chunks) - 1

    # previous_summaries: from all chunks before the last
    previous_summaries = [
        _chunk_summary_from_interactions(interactions_by_chunk[i])
        for i in range(last_idx)
    ]

    # current_summary: from the last chunk
    current_summary = _chunk_summary_from_interactions(interactions_by_chunk[last_idx])

    # current_objects: unified labels from the last chunk (mapping step)
    current_unified = mappings[last_idx].get("unified_objects", []) if last_idx < len(mappings) else []
    current_objects = sorted({
        (u.get("label") or "").lower()
        for u in current_unified
        if u.get("label")
    })

    # ---- Print the inputs as requested ----
    print("\n[predict] Inputs for predict_next_events()")
    print("[predict] previous_summaries:")
    for s in previous_summaries:
        print("  -", s)
    print("[predict] current_summary:", current_summary)
    print("[predict] current_objects:", current_objects)

    # ---- Run predictor ----
    NextEvents = predict_next_events(
        previous_summaries=previous_summaries,
        current_summary=current_summary,
        current_objects=current_objects,
        k=PREDICTIONS_TOP_K
    )

    # ---- Print the outputs as requested ----
    print("[predict] NextEvents:", NextEvents)

    # Stash into the final result for the frontend (easy to consume)
    pipeline_result["Predictions"] = {
        "last_chunk_index": last_idx,
        "previous_summaries": previous_summaries,
        "current_summary": current_summary,
        "current_objects": current_objects,
        "events": NextEvents,
        "model": PREDICTION_MODEL,
        "k": PREDICTIONS_TOP_K,
    }