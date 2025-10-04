import os
import re
import json
import tempfile
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------
# Optional: OpenCV metadata
# --------------------------
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

app = FastAPI(title="Video Interaction Connector", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ==========================
# Schemas
# ==========================
class ObjectItem(BaseModel):
    name: str = Field(..., example="car")
    description: Optional[str] = Field(None, example="blue car behind person; parked on the right")

class AnalyzeResponse(BaseModel):
    clip_id: str
    video_meta: dict
    objects: List[ObjectItem]
    relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    narrative: str

# ==========================
# Rule-based relation vocab
# ==========================
REL_MAP = {
    # spatial
    r"\b(left of|to the left of|on the left of|left)\b": "left_of",
    r"\b(right of|to the right of|on the right of|right)\b": "right_of",
    r"\b(behind|in the back of|at the back of)\b": "behind",
    r"\b(in front of|ahead of|infront of)\b": "in_front_of",
    r"\b(near|next to|beside|close to)\b": "near",
    r"\b(on top of|over|above)\b": "above",
    r"\b(under|below|beneath)\b": "below",
    r"\b(overlapping|overlap|covers)\b": "overlaps",

    # motion / interaction
    r"\b(approaching|walks? towards|moving towards|comes? closer to)\b": "approach",
    r"\b(following|follows)\b": "follow",
    r"\b(passing|passes)\b": "pass",
    r"\b(enters?|getting into|climbs? into)\b": "enter",
    r"\b(exits?|getting out of|climbs? out of|leaves?)\b": "exit",
    r"\b(opens?|opening)\b": "open",
    r"\b(closes?|closing)\b": "close",
    r"\b(holding|holds)\b": "hold",
    r"\b(giving|hands? to|passes? to)\b": "give",
    r"\b(takes?|takes from|picks? up)\b": "take",
    r"\b(riding|rides)\b": "ride",
    r"\b(driving|drives)\b": "drive",
    r"\b(run|running|jog|jogging)\b": "run",
    r"\b(walk|walking)\b": "walk",
    r"\b(stands?|standing)\b": "stand",
    r"\b(sits?|sitting)\b": "sit"
}

OBJECT_ALIASES = {
    "person": ["person", "man", "woman", "girl", "boy", "pedestrian", "human"],
    "car": ["car", "vehicle", "sedan", "taxi", "auto"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "bus": ["bus", "coach", "minibus"],
    "truck": ["truck", "lorry", "pickup"],
    "door": ["door"],
    "bag": ["bag", "backpack", "handbag"],
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
}
TOKEN_TO_CANON = {alias: canon for canon, toks in OBJECT_ALIASES.items() for alias in toks}

# ==========================
# Helpers (metadata, parsing)
# ==========================
def read_video_meta(file_bytes: bytes) -> dict:
    """Return light metadata from video (duration, fps) if OpenCV is available."""
    if not CV2_AVAILABLE:
        return {"fps": None, "duration_sec": None, "frames": None}
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {"fps": None, "duration_sec": None, "frames": None}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps > 0 else None
        cap.release()
        return {"fps": float(fps) if fps else None, "duration_sec": duration, "frames": frames}

def find_object_mentions(text: str, known_objects: List[str]) -> List[str]:
    """Return canonical object names mentioned in text (based on aliases + known names)."""
    text_l = text.lower()
    hits = set()
    for alias, canon in TOKEN_TO_CANON.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_l):
            hits.add(canon)
    for k in known_objects:
        if re.search(rf"\b{re.escape(k.lower())}\b", text_l):
            hits.add(k.lower())
    return list(hits)

# ==========================
# Rule-based extraction + narration
# ==========================
def extract_relations(objects: List[ObjectItem]) -> List[Tuple[str, str, str]]:
    """
    Very simple rule engine:
    - scan each object's description,
    - detect relation keywords,
    - bind (subject, relation, object) with mentioned objects.
    """
    names = [o.name.lower() for o in objects]
    desc_map = {o.name.lower(): (o.description or "") for o in objects}

    relations: List[Tuple[str, str, str]] = []
    for subj in names:
        desc = desc_map.get(subj, "")
        if not desc:
            continue
        for pattern, rel in REL_MAP.items():
            if re.search(pattern, desc, flags=re.IGNORECASE):
                mentioned = [m for m in find_object_mentions(desc, names) if m != subj]
                for obj in mentioned:
                    relations.append((subj, rel, obj))

    # Deduplicate
    dedup, seen = [], set()
    for t in relations:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup

def narrate(relations: List[Tuple[str, str, str]], objects: List[ObjectItem]) -> str:
    if not relations and objects:
        inventory = []
        for o in objects:
            inventory.append((o.description or o.name).strip().rstrip("."))
        return "; ".join(inventory) + "."
    if not relations:
        return "No clear interaction detected."

    verb_like = {
        "approach": "approaches", "enter": "enters", "exit": "exits",
        "open": "opens", "close": "closes", "follow": "follows", "pass": "passes",
        "ride": "rides", "drive": "drives", "run": "runs near", "walk": "walks near",
        "hold": "holds", "give": "gives to", "take": "takes from",
        "left_of": "is left of", "right_of": "is right of", "behind": "is behind",
        "in_front_of": "is in front of", "near": "is near", "above": "is above",
        "below": "is below", "overlaps": "overlaps with",
    }
    clauses, used = [], set()
    for (s, r, o) in relations[:5]:
        if (s, r, o) in used:
            continue
        used.add((s, r, o))
        clauses.append(f"{s} {verb_like.get(r, r)} {o}")
    sent = "; ".join(clauses)
    return sent[0].upper() + sent[1:] + "."

# ==========================
# (Optional) LLM Enhancer
# ==========================
# Uses OpenAI's Python SDK. Set OPENAI_API_KEY and (optionally) LLM_MODEL.
class LLMProvider:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    def infer(self, system: str, user: str, schema: dict, max_tokens: int = 600) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "VideoInteractionSchema", "schema": schema}},
                temperature=0.2,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

INTERACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subj": {"type": "string"},
                    "rel": {"type": "string"},
                    "obj": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"}
                },
                "required": ["subj", "rel", "obj"]
            }
        },
        "narrative": {"type": "string"}
    },
    "required": ["relations", "narrative"]
}

SYSTEM_PROMPT = (
    "You are a precise video scene-graph analyst. "
    "You take object names and natural-language descriptions for a short clip and infer interactions. "
    "Return only structured JSON that conforms to the provided schema. Be conservative and avoid hallucinating objects not present in inputs. "
    "Use short canonical relation verbs like: near, left_of, right_of, behind, in_front_of, overlaps, "
    "approach, enter, exit, open, close, follow, pass, ride, drive, hold, give, take, walk, run."
)

USER_PROMPT_TEMPLATE = """Inputs:
- Objects (name + description):
{objects_block}

- (Optional) Rule-based triples we already found:
{rules_block}

Task:
1) Normalize entity references (e.g., "man" -> "person", "sedan" -> "car").
2) Infer additional relations ONLY if strongly implied by the descriptions; avoid guessing.
3) Produce a concise one-sentence narrative summarizing what's happening.
4) Return strict JSON per the schema. Keep relations <= 8 and narrative <= 25 words.

Examples:
- If descriptions include "person approaching blue car", you can infer ["person","approach","car"].
- If descriptions say "car in front of person" and "person on the left of the car", include those spatial relations.

Now do it for the given inputs.
"""

def _build_user_prompt(objects: List[ObjectItem], rule_triples: List[Tuple[str, str, str]]) -> str:
    objs_txt = "\n".join([f"- {o.name}: {o.description or ''}" for o in objects]) or "(none)"
    rules_txt = "\n".join([f"- ({s}, {r}, {o})" for (s, r, o) in rule_triples]) or "(none)"
    return USER_PROMPT_TEMPLATE.format(objects_block=objs_txt, rules_block=rules_txt)

def _merge_relations(rule_triples: List[Tuple[str, str, str]], llm_relations: List[dict]) -> List[Tuple[str, str, str]]:
    out = set(rule_triples)
    for r in llm_relations or []:
        t = (r.get("subj", "").lower(), r.get("rel", "").lower(), r.get("obj", "").lower())
        if all(t):
            out.add(t)
    return list(out)

def llm_enhance(objects: List[ObjectItem], rule_triples: List[Tuple[str, str, str]]) -> Tuple[List[Tuple[str, str, str]], str]:
    if not os.getenv("OPENAI_API_KEY"):
        # if no key, just return rule-based
        return rule_triples, ""
    provider = LLMProvider()
    user_prompt = _build_user_prompt(objects, rule_triples)
    llm_json = provider.infer(SYSTEM_PROMPT, user_prompt, INTERACTION_SCHEMA)
    merged = _merge_relations(rule_triples, llm_json.get("relations", []))
    narrative = (llm_json.get("narrative") or "").strip()
    return merged, narrative

# ==========================
# Endpoint
# ==========================
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    videoclip: UploadFile = File(..., description="Short video clip file (e.g., mp4)"),
    objects_json: str = Form(..., description='JSON list of {"name": str, "description": str}'),
    use_llm: bool = Form(False, description="Set true to enhance with an LLM (if OPENAI_API_KEY set)")
):
    """
    Input:
      - videoclip: the mini-clip file (we keep/inspect minimal meta)
      - objects_json: JSON array with items like:
          [{"name":"person","description":"person walking towards blue car"},
           {"name":"car","description":"blue car in front of person"}]
      - use_llm: optional toggle to enhance results via an LLM
    Output:
      - clip_id, video_meta, objects, relations (triples), narrative (text)
    """
    file_bytes = await videoclip.read()
    clip_id = os.path.splitext(videoclip.filename or "clip")[0]

    # Optional video metadata
    video_meta = read_video_meta(file_bytes)

    # Parse objects
    try:
        raw = json.loads(objects_json)
        objects = [ObjectItem(**item) for item in raw]
    except Exception as e:
        return {
            "clip_id": clip_id,
            "video_meta": video_meta,
            "objects": [],
            "relations": [],
            "narrative": f"Invalid objects_json: {e}"
        }

    # Rule-based baseline
    rule_rel = extract_relations(objects)
    rule_narr = narrate(rule_rel, objects)

    # Optional LLM enhancement
    if use_llm:
        try:
            merged_rel, llm_narr = llm_enhance(objects, rule_rel)
            final_rel = merged_rel
            final_narr = llm_narr or rule_narr
        except HTTPException as e:
            # graceful fallback
            final_rel, final_narr = rule_rel, f"{rule_narr} (LLM unavailable)"
    else:
        final_rel, final_narr = rule_rel, rule_narr

    return AnalyzeResponse(
        clip_id=clip_id,
        video_meta=video_meta,
        objects=objects,
        relations=final_rel,
        narrative=final_narr
    )
