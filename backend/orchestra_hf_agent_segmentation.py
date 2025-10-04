"""
Agent + HF Segmentation integration (single-chunk)

Main entry:
    map_chunk_with_hf_and_agent(
        clip: cvclip.CVClip,
        seg_processor,
        seg_model,
        describer_agent_call,           # callable(prompt:str, images_b64:list[str]) -> str(JSON)
        describer_prompt: str,
        client,                          # OpenAI client for mapping call
        mapping_model: str = "gpt-4o",
        frames_for_describer: int = 4,
        max_width: int = 640,
        min_pixels: int = 200,
        keep_labels: list[str] | None = None,
        k_neighbors: int = 5,
    ) -> dict

What it does:
    - For ONE chunk:
        * Samples a few frames and calls your vision describer agent (objects/relations JSON)
        * Runs HF SegFormer on the middle frame and builds instance segments
        * Builds a payload with precise geometry + neighbors
        * Calls the mapping LLM to reconcile describer objects ↔ segments
    - Returns the mapping JSON.

Inputs:
    clip                : single CVClip
    seg_processor/model : from hf_semantic_segmentation.load_segformer(...)
    describer_agent_call: callable prompt+images -> JSON string
    describer_prompt    : your ON_SCREEN_ONLY_PROMPT (or custom)
    client              : OpenAI client used only for the mapping call
    mapping_model       : LLM name for mapping
    frames_for_describer: number of frames to send to describer
    max_width           : resize width for frames
    min_pixels          : area filter for instance blobs
    keep_labels         : optional whitelist of labels for the HF segments
    k_neighbors         : neighbor count for geometry payload

Output:
    dict: {"unified_objects":[...], "unmatched": {...}} from the mapping LLM

Usage:
    from agent_hf_integration import map_chunk_with_hf_and_agent
    mapping = map_chunk_with_hf_and_agent(
        clip, seg_processor, seg_model, describer_agent_call, ON_SCREEN_ONLY_PROMPT, client
    )
"""

from typing import List, Dict, Any, Tuple, Callable
import json
import io, base64
from PIL import Image
from typing import List, Dict, Any, Tuple, Callable, Optional
# import segmenter util
from hf_segmentation import segments_from_chunk

# Use our OpenCV-based clip shim
from cvclip import CVClip

# -------------------------- small frame helpers --------------------------

def _sample_times(duration: float, frames_per_chunk: int) -> List[float]:
    if duration <= 0 or frames_per_chunk <= 0:
        return [0.0]
    step = duration / float(frames_per_chunk + 1)
    return [step * (i + 1) for i in range(frames_per_chunk)]


def _clip_frames_to_b64(
    clip: CVClip, times: List[float], max_width: int = 640
) -> List[str]:
    images_b64: List[str] = []
    for t in times:
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        if max_width and img.width > max_width:
            h = int(img.height * (max_width / float(img.width)))
            img = img.resize((max_width, h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images_b64.append(f"data:image/png;base64,{b64}")
    return images_b64


# ---------------------------- mapping helpers ----------------------------

import math
import numpy as np

def _box_center(b: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _box_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / max(1, union))

def _dir_from_delta(dx: float, dy: float, dead: float = 0.02) -> str:
    s = []
    if abs(dy) > dead:
        s.append("above" if dy < 0 else "below")
    if abs(dx) > dead:
        s.append("left_of" if dx < 0 else "right_of")
    return "+".join(s) if s else "overlapping_or_same_spot"

def _normalize_geometry(bbox: List[int], centroid: List[float], W: int, H: int) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    cx, cy = centroid
    return {
        "bbox_abs": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_norm": [x1 / W, y1 / H, x2 / W, y2 / H],
        "centroid_abs": [float(cx), float(cy)],
        "centroid_norm": [cx / W, cy / H],
        "w_norm": w / W,
        "h_norm": h / H,
        "area_norm": (w * h) / float(W * H),
    }

def _build_neighbors(segments_api: List[Dict[str, Any]], W: int, H: int, k_neighbors: int = 5) -> Dict[int, List[Dict[str, Any]]]:
    diag = math.sqrt(W * W + H * H)
    boxes = { int(s["seg_id"]): s["bbox"] for s in segments_api }
    cents = { int(s["seg_id"]): (float(s["centroid"][0]), float(s["centroid"][1])) for s in segments_api }
    labels= { int(s["seg_id"]): s["label"] for s in segments_api }

    neighbors = {}
    for a_id, a_c in cents.items():
        arr = []
        for b_id, b_c in cents.items():
            if b_id == a_id:
                continue
            dx = (b_c[0] - a_c[0]) / W
            dy = (b_c[1] - a_c[1]) / H
            dist = math.sqrt((b_c[0] - a_c[0])**2 + (b_c[1] - a_c[1])**2) / diag
            iou = _box_iou(boxes[a_id], boxes[b_id])
            arr.append({
                "neighbor_id": int(b_id),
                "label": labels[b_id],
                "distance_norm": float(dist),
                "dx": float(dx),
                "dy": float(dy),
                "direction": _dir_from_delta(dx, dy),
                "iou": float(iou),
            })
        arr.sort(key=lambda d: (d["distance_norm"], -d["iou"]))
        neighbors[a_id] = arr[:k_neighbors]
    return neighbors


def build_mapping_payload_v2(
    objects_agent: List[Dict[str, Any]],
    segments_api: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    k_neighbors: int = 5,
) -> Dict[str, Any]:
    W, H = image_size

    objs_norm = []
    for o in objects_agent:
        objs_norm.append({
            "id": o.get("id"),
            "name": (o.get("name") or "").strip(),
            "confidence": float(o.get("confidence", 0.0)),
            "attributes": (o.get("attributes") or "").strip(),
        })

    segs_norm = []
    for s in segments_api:
        geo = _normalize_geometry(s["bbox"], s["centroid"], W, H)
        segs_norm.append({
            "seg_id": int(s["seg_id"]),
            "label": (s["label"] or "").strip(),
            **geo
        })

    neigh = _build_neighbors(segments_api, W, H, k_neighbors=k_neighbors)

    return {
        "image_meta": {"width": int(W), "height": int(H), "diag_norm": 1.0},
        "describer_objects": objs_norm,
        "segmenter_objects": segs_norm,
        "segmenter_neighbors": neigh
    }


MAPPING_SYSTEM_PROMPT_V2 = """
You unify objects from two sources for a SINGLE image:

SOURCE A (DESCRIBER):
- objects: {id, name, confidence, attributes}

SOURCE B (SEGMENTER):
- objects: {seg_id, label, bbox_abs, bbox_norm, centroid_abs, centroid_norm, w_norm, h_norm, area_norm}
- neighbors: For each seg_id, top-k nearest neighbors with precise numbers:
  {neighbor_id, label, distance_norm (0..1 by image diagonal), dx, dy (normalized), direction (e.g., 'left_of+above'), iou}

TASK:
- Produce a one-to-one (or best-effort) mapping between DESCRIBER objects and SEGMENTER objects.
- Resolve synonyms/hypernyms WITHOUT hardcoded dictionaries (infer linguistically).
- Use DESCRIBER attributes plus numeric geometry to disambiguate.
- If no plausible match exists, leave that side unmatched.

OUTPUT strictly the JSON:
{
  "unified_objects": [
    {
      "object_id": "string",             // DESCRIBER id if matched; else "seg_<seg_id>"
      "source_ids": {
        "describer_id": "string|null",
        "segment_id": "int|null"
      },
      "label": "string",
      "synonyms_considered": ["string", ...],
      "attributes": "string",
      "bbox": [x1, y1, x2, y2] | null,
      "centroid": [cx, cy] | null,
      "match_confidence": 0.0,
      "notes": "brief rationale"
    }
  ],
  "unmatched": {
    "describer_only": ["id", ...],
    "segmenter_only": [seg_id, ...]
  }
}
Return ONLY valid JSON.
"""


def call_openai_json(client, system_prompt: str, payload: dict, model: str = "gpt-4o", temperature: float = 0.0):
    """
    Calls the LLM and guarantees a JSON object back.
    """
    msg_explainer = (
        "SOURCE A objects are under 'describer_objects'. "
        "SOURCE B objects are under 'segmenter_objects'. "
        "Neighbors are under 'segmenter_neighbors'. "
        "Image size is in 'image_meta'.\n\n"
        "Here is the JSON payload:"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": msg_explainer},
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
            ]}
        ]
    )
    return json.loads(resp.choices[0].message.content)


def map_objects_via_llm_v2(
    client,
    objects_agent: List[Dict[str, Any]],
    segments_api: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    k_neighbors: int = 5,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    payload = build_mapping_payload_v2(objects_agent, segments_api, image_size, k_neighbors=k_neighbors)
    result = call_openai_json(client, MAPPING_SYSTEM_PROMPT_V2, payload, model=model)
    if not isinstance(result, dict) or "unified_objects" not in result:
        raise ValueError("LLM returned unexpected shape.")
    return result


# -------------------- Public API: single-chunk integration --------------------

def map_chunk_with_hf_and_agent(
    clip: CVClip,
    seg_processor,
    seg_model,
    describer_agent_call: Callable[[str, List[str]], str],
    describer_prompt: str,
    client,
    mapping_model: str = "gpt-4o",
    frames_for_describer: int = 4,
    max_width: int = 640,
    min_pixels: int = 200,
    keep_labels: Optional[List[str]] = None,
    k_neighbors: int = 5,
) -> Dict[str, Any]:
    """
    One-chunk pipeline:
      - calls agentic describer over several frames
      - runs HF segmenter on middle frame
      - maps objects ↔ segments via LLM
    Returns mapping JSON.
    """
    # 1) Agentic describer on a few frames
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    times = _sample_times(duration, frames_for_describer)
    images_b64 = _clip_frames_to_b64(clip, times, max_width=max_width)

    raw_desc = describer_agent_call(describer_prompt, images_b64)
    try:
        parsed_desc = json.loads(raw_desc)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw_desc, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Describer did not return valid JSON.\nGot:\n{raw_desc}")
        parsed_desc = json.loads(m.group(0))

    objects_agent = parsed_desc.get("objects", [])

    # 2) HF segments on middle frame
    segments_api, (W, H) = segments_from_chunk(
        clip=clip,
        seg_processor=seg_processor,
        seg_model=seg_model,
        max_width=max_width,
        min_pixels=min_pixels,
        keep_labels=keep_labels,
        rle_order="C",
    )

    # 3) Mapping
    mapping = map_objects_via_llm_v2(
        client=client,
        objects_agent=objects_agent,
        segments_api=segments_api,
        image_size=(W, H),
        k_neighbors=k_neighbors,
        model=mapping_model,
    )
    return mapping


__all__ = [
    "map_chunk_with_hf_and_agent",
    "map_objects_via_llm_v2",
    "build_mapping_payload_v2",
]
