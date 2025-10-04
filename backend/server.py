# server.py
import os, uuid, tempfile
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ===== import your pipeline helpers =====
from chunker import chunk_video, close_video_chunks             # motion-based splitter (CVClip subclips)
from cvclip import CVClip                                       # lightweight MoviePy-like shim
from agent_segmentation import (                                # LLM describer (objects/relations)
    describe_chunks_on_screen,
    ON_SCREEN_ONLY_PROMPT,
    build_agent_call,
)
from hf_segmentation import (                                   # HF SegFormer (single chunk)
    load_segformer,
    segments_from_chunk,
)
from orchestra_hf_agent_segmentation import (                   # LLM mapping (agent ↔ HF)
    map_objects_via_llm_v2,
)
from agent_spec_captioning import (                             # Interactions JSON (single chunk)
    set_openai_client,
    describe_interactions_for_labels,
)

# ================= FastAPI boilerplate =================
app = FastAPI(title="Video Scene Graph API", version="1.0")

# CORS: loosen for local dev; tighten for prod (specific domain/ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g., ["http://localhost:5173", "http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== Response models ===============
class ChunkOut(BaseModel):
    id: str
    startMs: int
    endMs: int

class ChunkListOut(BaseModel):
    uploadId: str
    chunks: List[ChunkOut]

class AnalysisOut(BaseModel):
    narrative: str
    relations: List[List[str]]              # [label1, label2, description]
    objects: List[Dict[str, str]]           # {name, description?}
    predictions: Optional[List[Dict[str, Any]]] = None  # optional

# =============== Config / Globals ===============
# OpenAI client (prefer env var OPENAI_API_KEY)
client = OpenAI()  # or OpenAI(api_key="sk-proj-...") for local-only testing
set_openai_client(client)  # required once by interactions module

# Describer (vision)
DESCRIBER_MODEL = "gpt-4o"        # or "gpt-4.1-mini" / "gpt-4o-mini"
FRAMES_PER_CHUNK = 4
MAX_FRAME_WIDTH = 640

# HF SegFormer
SEGFORMER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
seg_processor, seg_model = load_segformer(SEGFORMER_ID)  # load once at boot

# Mapping
MAPPING_MODEL = "gpt-4o"
K_NEIGHBORS = 5

# Interactions
INTERACTION_MODEL = "gpt-4o"
N_INTERACTION_FRAMES = 2

# Prediction (optional, next-chunk guess)
PREDICTION_MODEL = "gpt-4.1-mini"
PREDICTIONS_TOP_K = 3

# Store upload sessions: where the video lives and each chunk's [start,end]
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Build describer call once
agent_call = build_agent_call(client, model=DESCRIBER_MODEL)


# =============== Small helpers ===============
def _chunk_summary_from_interactions(inter_json: Dict[str, Any]) -> str:
    """
    Compact narrative from interactions JSON.
    """
    interactions = inter_json.get("interactions", []) if inter_json else []
    if not interactions:
        return "No interactions"
    parts = []
    for it in interactions[:8]:
        labs = it.get("labels") or []
        action = (it.get("action") or "").strip()
        desc = (it.get("description") or "").strip()
        if action and len(labs) >= 2:
            parts.append(f"{labs[0]} {action} {labs[1]}")
        elif desc:
            parts.append(desc)
    return "; ".join(parts) if parts else "No interactions"


def predict_next_events(previous_summaries: List[str], current_summary: str, current_objects: List[str], k: int = 3) -> List[str]:
    """
    Simple LLM predictor for likely next events (strings, one per line).
    """
    prompt = f"""
You are an action prediction model for a video-to-graph system.

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


# ======================= Endpoints =======================

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/chunk-video", response_model=ChunkListOut)
async def chunk_video_endpoint(file: UploadFile = File(...)):
    """
    Accepts a video file, writes to temp, splits into motion-based chunks,
    returns IDs and boundaries (ms) for the FE timeline cards.
    """
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Use your motion-based splitter -> CVClip subclips (remember to close them)
    clips = chunk_video(tmp_path)  # :contentReference[oaicite:7]{index=7}
    try:
        boundaries: List[Tuple[float, float]] = []
        chunks_out: List[ChunkOut] = []
        upload_id = uuid.uuid4().hex

        for i, c in enumerate(clips):
            start_s = float(getattr(c, "_start", 0.0))
            end_s   = float(getattr(c, "_end",   start_s + getattr(c, "duration", 0.0)))
            boundaries.append((start_s, end_s))
            chunks_out.append(ChunkOut(
                id=f"{upload_id}:{i}",
                startMs=int(round(start_s * 1000)),
                endMs=int(round(end_s   * 1000)),
            ))
    finally:
        close_video_chunks(clips)  # free file handles

    SESSIONS[upload_id] = {
        "video_path": tmp_path,
        "boundaries": boundaries,
    }
    return ChunkListOut(uploadId=upload_id, chunks=chunks_out)


@app.get("/api/analysis/{chunk_id}", response_model=AnalysisOut)
def analyze_chunk_endpoint(chunk_id: str):
    """
    Returns exactly what the FE graph modal expects:
      - narrative: string
      - relations: [ [label1, label2, description], ... ]
      - objects  : [ {name, description?}, ... ]
      - predictions (optional): [{event, probability, description}]
    """
    if ":" not in chunk_id:
        raise HTTPException(status_code=400, detail="Invalid chunk id.")
    upload_id, idx_str = chunk_id.split(":", 1)
    if upload_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Upload session not found.")
    try:
        idx = int(idx_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chunk index.")

    sess = SESSIONS[upload_id]
    video_path = sess["video_path"]
    boundaries = sess["boundaries"]
    if idx < 0 or idx >= len(boundaries):
        raise HTTPException(status_code=404, detail="Chunk index out of range.")

    start_s, end_s = boundaries[idx]

    # Work on this chunk
    clip_full = CVClip(video_path)                                            # :contentReference[oaicite:8]{index=8}
    clip = clip_full.subclip(start_s, end_s)

    # 1) Describer (vision LLM) on several frames
    desc_list = describe_chunks_on_screen(                                    # :contentReference[oaicite:9]{index=9}
        video_chunks=[clip],
        agent_call=agent_call,
        prompt=ON_SCREEN_ONLY_PROMPT,
        frames_per_chunk=FRAMES_PER_CHUNK,
        max_width=MAX_FRAME_WIDTH,
    )
    agent_json = desc_list[0]["json"]
    objects_agent = agent_json.get("objects", [])

    # 2) HF SegFormer on middle frame of this chunk
    segs, (W, H) = segments_from_chunk(                                      # :contentReference[oaicite:10]{index=10}
        clip=clip,
        seg_processor=seg_processor,
        seg_model=seg_model,
        max_width=MAX_FRAME_WIDTH,
        min_pixels=200,
        keep_labels=None,
        rle_order="C",
    )

    # 3) Mapping (objects ↔ segments)
    mapping = map_objects_via_llm_v2(                                        # :contentReference[oaicite:11]{index=11}
        client=client,
        objects_agent=objects_agent,
        segments_api=segs,
        image_size=(W, H),
        k_neighbors=K_NEIGHBORS,
        model=MAPPING_MODEL,
    )
    unified = mapping.get("unified_objects", [])

    # 4) Interactions (action-level pairs)
    mapping_with_meta = {**mapping, "meta": {"image_size": [W, H]}}
    interactions = describe_interactions_for_labels(                          # :contentReference[oaicite:12]{index=12}
        clip=clip,
        mapping_chunk=mapping_with_meta,
        include_labels=sorted({(u.get("label") or "").lower() for u in unified if u.get("label")}),
        n_frames=N_INTERACTION_FRAMES,
        model=INTERACTION_MODEL,
    )

    # ---- Pack for the FE ----
    # narrative
    narrative = _chunk_summary_from_interactions(interactions)

    # relations: [label1, label2, description]
    relations: List[List[str]] = []
    for it in interactions.get("interactions", []):
        labs = it.get("labels") or []
        desc = (it.get("description") or "").strip()
        if len(labs) >= 2:
            relations.append([str(labs[0]).lower(), str(labs[1]).lower(), desc])

    # objects: from mapping unified objects; attributes → description (if any)
    objects: List[Dict[str, str]] = []
    for u in unified:
        label = (u.get("label") or "").strip()
        if not label:
            continue
        desc_attr = (u.get("attributes") or "").strip()
        objects.append({"name": label, **({"description": desc_attr} if desc_attr else {})})

    # predictions for NEXT chunk (optional, used by FE to draw dashed edges if desired)
    # gather a few previous chunk narratives for context
    previous_summaries: List[str] = []
    for j in range(max(0, idx - 3), idx):  # only last few to keep latency small
        ps, pe = boundaries[j]
        prev_clip = clip_full.subclip(ps, pe)
        # quick interaction pass to derive a narrative summary
        prev_inter = describe_interactions_for_labels(
            clip=prev_clip,
            mapping_chunk={"unified_objects": unified, "meta": {"image_size": [W, H]}},
            include_labels=sorted({o["name"] for o in objects}),
            n_frames=1,
            model=INTERACTION_MODEL,
        )
        previous_summaries.append(_chunk_summary_from_interactions(prev_inter))

    current_objects = sorted({o["name"] for o in objects})
    next_events = predict_next_events(previous_summaries, narrative, current_objects, k=PREDICTIONS_TOP_K)
    predictions = [{"event": p, "probability": 0.5, "description": p} for p in next_events]

    # cleanup
    clip.close()
    clip_full.close()

    return {
        "narrative": narrative,
        "relations": relations,
        "objects": objects,
        "predictions": predictions,
    }
