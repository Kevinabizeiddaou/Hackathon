"""
Interaction Extractor (single-chunk)

Main entry:
    describe_interactions_for_labels(
        clip: cvclip.CVClip,
        mapping_chunk: dict,
        include_labels: list[str],
        n_frames: int = 3,
        model: str = "gpt-4o",
    ) -> dict

What it does:
    - Samples 1–3 frames from ONE video chunk.
    - Builds a compact JSON of ONLY the allowed/mapped entities (+ lightweight pairwise cues).
    - Calls an LLM (vision) to return ONLY action-level interactions JSON.

Inputs:
    clip           : A single CVClip (the chunk to analyze)
    mapping_chunk  : Mapping JSON for this chunk (from your orchestrator/mapper)
                     Expected keys (as produced by your pipeline):
                       - "unified_objects": [
                           {"object_id": str, "label": str, "attributes": str,
                            "bbox": [x1,y1,x2,y2] | null, "centroid": [cx,cy] | null, ...}, ...
                         ]
                       - Optional: "meta": {"image_size": [W,H]}
    include_labels : Only these labels are considered as candidates for interactions
    n_frames       : How many frames to sample for the model (1–3 recommended)
    model          : OpenAI model name for the JSON interaction call

Output:
    dict:
      {
        "interactions": [
          {
            "labels": ["label_a","label_b"],
            "object_ids": ["id_a","id_b"],
            "action": "canonical_verb_lemma",
            "description": "short present-tense sentence",
            "confidence": 0.0
          },
          ...
        ]
      }

Setup:
    from openai import OpenAI
    from interactions_extractor import set_openai_client, describe_interactions_for_labels

    client = OpenAI()
    set_openai_client(client)  # required once per process

    result = describe_interactions_for_labels(
        clip=chunk,
        mapping_chunk=mapping_json_for_this_chunk,
        include_labels=["person","animal","apple"],
        n_frames=3,
        model="gpt-4o"
    )
"""

from typing import List, Dict, Any, Tuple
import io, json, base64, math
from PIL import Image
import numpy as np

# Use our lightweight OpenCV-based clip
from cvclip import CVClip


# ------------------ REQUIRED: set OpenAI client once ------------------

# We'll keep your function signatures intact and rely on a module-global client,
# but expose a setter so importing this file does not execute anything.
client = None

def set_openai_client(openai_client) -> None:
    """
    Set the OpenAI client used by this module.

    Usage:
        from openai import OpenAI
        from interactions_extractor import set_openai_client
        set_openai_client(OpenAI())
    """
    global client
    client = openai_client


# ---------------------------- System prompt ----------------------------

INTERACTION_JSON_SYSTEM_PROMPT = """You analyze ONE video chunk and ONLY report
action-level interactions (verbs) between entities whose labels are INCLUDED.

Inputs:
- 1–3 frames from the chunk.
- A compact JSON of ONLY allowed entities (object_id, label, attributes, bbox, centroid).
- Optional pairwise numeric cues (distance_norm, iou, direction).

Output: ONLY JSON
{
  "interactions": [
    {
      "labels": ["label_a","label_b"],        // lowercase, human-readable
      "object_ids": ["id_a","id_b"],          // if known
      "action": "canonical_verb_lemma",       // one from the allowed list below
      "description": "short present-tense sentence about these two",  // e.g., "person is feeding the animal"
      "confidence": 0.0                       // 0..1
    }
  ]
}

Hard constraints:
- Report ONLY if an ACTION is visible or strongly implied by pixels.
- REJECT purely spatial/static relations: ban "near", "next to", "beside", "above", "below",
  "in front of", "behind", "around", "close to".
- 'action' MUST be one (case-insensitive) of:
  holding, handing, giving, taking, feeding, eating, drinking, biting, licking, sniffing,
  touching, patting, petting, pointing, pulling, pushing, carrying, throwing, catching,
  riding, mounting, dismounting, leading, opening, closing, jumping, jumping_over, kicking,
  reading, writing, approaching, shaking, waving
- 'description' must be ≤ 16 words, present tense, pixel-grounded, and mention ONLY the two labels.
- If no valid actions, return {"interactions": []}.
- No extra text outside JSON."""


# ------------------------ Low-level OpenAI call ------------------------

def call_openai_json_with_images(
    system_prompt: str,
    user_text: str,
    image_urls: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.1
) -> dict:
    """
    Call the LLM with image URLs (data URLs OK) and return a JSON object.

    NOTE: Requires set_openai_client(...) to have been called beforehand.
    """
    if client is None:
        raise RuntimeError("OpenAI client not set. Call set_openai_client(OpenAI()) first.")

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            [{"type":"text","text": user_text}] +
            [{"type":"image_url","image_url":{"url": u}} for u in image_urls]
        )},
    ]
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=msgs,
    )
    return json.loads(resp.choices[0].message.content)


# ----------------------------- Frame utils -----------------------------

def _frames_as_b64_list(clip: CVClip, n_frames: int = 3, max_width: int = 640) -> List[str]:
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    if duration <= 0 or n_frames <= 0:
        times = [0.0]
    else:
        step = duration / (n_frames + 1)
        times = [step * (i + 1) for i in range(n_frames)]
    urls = []
    for t in times:
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        if max_width and img.width > max_width:
            h = int(img.height * (max_width / float(img.width)))
            img = img.resize((max_width, h), Image.BILINEAR)
        buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        urls.append(f"data:image/png;base64,{b64}")
    return urls


# --------------------------- Geometry helpers --------------------------

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
    """
    Cardinal-ish direction using small dead-zone; expects dx,dy normalized by (W,H).
    """
    s = []
    if abs(dy) > dead:
        s.append("above" if dy < 0 else "below")
    if abs(dx) > dead:
        s.append("left_of" if dx < 0 else "right_of")
    return "+".join(s) if s else "overlapping_or_same_spot"


def _pairwise_features(objs: List[dict], W: int, H: int) -> List[dict]:
    """
    Build lightweight numeric cues between candidates (if bbox/centroid exist).
    """
    pairs = []
    diag = (W**2 + H**2) ** 0.5
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            a, b = objs[i], objs[j]
            aid, bid = str(a["object_id"]), str(b["object_id"])
            lab_a, lab_b = (a.get("label") or "").lower(), (b.get("label") or "").lower()
            ca, cb = a.get("centroid"), b.get("centroid")
            ba, bb = a.get("bbox"), b.get("bbox")
            dist_norm = None; iou = None; direction = "unknown"
            if ca and cb:
                dx = (cb[0]-ca[0]) / max(1, W)
                dy = (cb[1]-ca[1]) / max(1, H)
                dist_norm = ((cb[0]-ca[0])**2 + (cb[1]-ca[1])**2) ** 0.5 / max(1.0, diag)
                direction = _dir_from_delta(dx, dy)
            if ba and bb:
                iou = _box_iou(ba, bb)
            pairs.append({
                "a": {"id": aid, "label": lab_a},
                "b": {"id": bid, "label": lab_b},
                "distance_norm": None if dist_norm is None else float(dist_norm),
                "iou": None if iou is None else float(iou),
                "direction": direction
            })
    return pairs


# -------------------------- Public API (single-chunk) --------------------------

def describe_interactions_for_labels(
    clip: CVClip,
    mapping_chunk: dict,
    include_labels: List[str],
    n_frames: int = 3,
    model: str = "gpt-4o"
) -> dict:
    """
    Returns ONLY the interaction JSON for ONE chunk:
      {"interactions":[
         {"labels":["person","apple"], "object_ids":["2","3"], "action":"feeding",
          "description":"person is feeding the animal", "confidence":0.87},
         ...
      ]}

    - Only mentions pairs whose labels are in `include_labels` AND judged interacting.
    - Uses mapping_chunk["unified_objects"] to fetch object_id/label/attributes/bbox/centroid.
    """
    allowed = {s.lower().strip() for s in include_labels}

    # Filter mapping to ONLY allowed labels
    focus_objs = []
    for u in mapping_chunk.get("unified_objects", []):
        lab = (u.get("label") or "").lower()
        if lab in allowed:
            focus_objs.append({
                "object_id": str(u.get("object_id")),
                "label": (u.get("label") or ""),
                "attributes": (u.get("attributes") or ""),
                "bbox": u.get("bbox"),
                "centroid": u.get("centroid"),
            })

    # If <2 focus objects, nothing to consider
    if len(focus_objs) < 2:
        return {"interactions": []}

    # Numeric cues for pairs
    W = mapping_chunk.get("meta", {}).get("image_size", [None, None])[0]
    H = mapping_chunk.get("meta", {}).get("image_size", [None, None])[1]
    # If meta is missing, estimate from largest bbox we see
    if not W or not H:
        xs = []; ys = []
        for o in focus_objs:
            b = o.get("bbox")
            if b:
                xs += [b[0], b[2]]
                ys += [b[1], b[3]]
        if xs and ys:
            W = max(xs) + 1
            H = max(ys) + 1
        else:
            W, H = 640, 480  # fallback to avoid div-by-zero

    pair_feats = _pairwise_features(focus_objs, W, H)

    # Build compact JSON for the model
    compact = {
        "allowed_labels": sorted(list(allowed)),
        "focus_objects": focus_objs,
        "pairwise": pair_feats
    }

    # Sample frames
    imgs = _frames_as_b64_list(clip, n_frames=n_frames, max_width=640)

    # Ask the model for JSON ONLY (no extra text)
    user_text = (
      "Return ONLY action-level interactions using the allowed verbs, with both 'action' and a short 'description'. "
      "Ignore spatial-only relations. If none, return an empty list.\n\n"
      f"FOCUS JSON:\n{json.dumps(compact, ensure_ascii=False)}"
    )

    out = call_openai_json_with_images(
        system_prompt=INTERACTION_JSON_SYSTEM_PROMPT,
        user_text=user_text,
        image_urls=imgs,
        model=model,
        temperature=0.1
    )

    # Light post-check to ensure schema keys exist
    if not isinstance(out, dict) or "interactions" not in out or not isinstance(out["interactions"], list):
        return {"interactions": []}

    # Normalize labels to lower for consistency
    for it in out["interactions"]:
        if "labels" in it and isinstance(it["labels"], list):
            it["labels"] = [str(x).lower() for x in it["labels"]]
    return out


__all__ = [
    "set_openai_client",
    "describe_interactions_for_labels",
    "INTERACTION_JSON_SYSTEM_PROMPT",
]
