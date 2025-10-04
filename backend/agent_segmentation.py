"""
Agent-based chunk describer (objects + optional relations).

Main entry:
    describe_chunks_on_screen(
        video_chunks: list[cvclip.CVClip],
        agent_call: Callable[[str, list[str]], str],
        prompt: str,
        frames_per_chunk: int = 4,
        max_width: int = 640,
        temperature: float = 0.0,
    ) -> list[dict]

What it does:
    For each chunk, samples a few frames, encodes them as base64 data URLs, and
    calls a provided LLM vision agent. Collects the STRICT-JSON response per chunk.

Inputs:
    video_chunks: list of CVClip chunks (from chunker.chunk_video)
    agent_call: a callable(prompt: str, images_b64: list[str]) -> str (JSON string)
    prompt: the instruction JSON schema (e.g., ON_SCREEN_ONLY_PROMPT)
    frames_per_chunk: how many frames to sample per chunk
    max_width: optional downscale width to control token cost
    temperature: (not used internally; pass through if your agent_call uses it)

Output:
    A list of dicts like:
      [{"chunk_index": int, "json": dict, "raw": str}, ...]

Convenience:
    Use `build_agent_call(client, model="gpt-4o")` to create an `agent_call`
    compatible with OpenAI's Chat Completions API (responses enforced as JSON).
"""

from typing import List, Dict, Any, Callable
import io, base64, json
from PIL import Image

# swapped MoviePy for our lightweight OpenCV shim
from cvclip import CVClip


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


def describe_chunks_on_screen(
    video_chunks: List[CVClip],
    agent_call: Callable[[str, List[str]], str],
    prompt: str,
    frames_per_chunk: int = 4,
    max_width: int = 640,
    temperature: float = 0.0,  # kept for signature parity; agent_call may ignore/use it
) -> List[Dict[str, Any]]:
    """
    Calls a JSON-enforced vision agent per chunk and returns parsed + raw outputs.
    """
    results: List[Dict[str, Any]] = []

    for idx, clip in enumerate(video_chunks):
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        times = _sample_times(duration, frames_per_chunk)
        images_b64 = _clip_frames_to_b64(clip, times, max_width=max_width)

        raw = agent_call(prompt, images_b64)

        try:
            parsed = json.loads(raw)
        except Exception:
            # Backup: attempt to extract the first {...} block
            import re
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                raise ValueError(
                    f"Agent did not return valid JSON for chunk {idx}. Got:\n{raw}"
                )
            parsed = json.loads(m.group(0))

        results.append({"chunk_index": idx, "json": parsed, "raw": raw})

    return results


# ---------------------- Prompt & agent_call helper ----------------------

ON_SCREEN_ONLY_PROMPT = """You are a vision agent that ONLY describes what is visible in the image.
Do NOT infer intent, brand names, or off-screen causes. Be literal and constrained to pixels.

Return ONLY a valid compact JSON object with this exact schema:

{
  "objects": [
    {
      "id": "unique_id_for_object",
      "name": "person|car|table|... (generic visible objects)",
      "confidence": 0.0,
      "attributes": "brief positional description (e.g., 'bus next to person on the right', 'person behind another person')"
    }
  ],
  "relations": [
    {
      "subject_id": "id of object A",
      "predicate": "left_of|right_of|in_front_of|behind|overlapping|on|under|holding|riding|near",
      "object_id": "id of object B",
      "confidence": 0.0
    }
  ]
}

Rules:
- objects: include only clearly visible items; confidence ∈ [0,1]. Make ids stable within a chunk (e.g., 'person_1', 'horse_2').
- attributes: must be a short visible spatial description tied to pixels.
- relations: only include if visually supportable in the frame(s). Use 'near' only when objects are close relative to their sizes.
- If no relations are visible, return "relations": [].
- Output ONLY valid JSON. No extra text.
"""


def build_agent_call(client, model: str = "gpt-4o") -> Callable[[str, List[str]], str]:
    """
    Returns a callable(prompt, images_b64) -> raw_json_str using OpenAI's Chat Completions.
    Enforces JSON response format.
    """
    def _call(prompt: str, imgs: List[str]) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You output ONLY valid JSON matching the user's schema."},
                {
                    "role": "user",
                    "content": (
                        [{"type": "text", "text": prompt}]
                        + [{"type": "image_url", "image_url": {"url": url}} for url in imgs]
                    ),
                },
            ],
        )
        return resp.choices[0].message.content
    return _call


__all__ = [
    "describe_chunks_on_screen",
    "ON_SCREEN_ONLY_PROMPT",
    "build_agent_call",
]
