"""
Hugging Face Semantic Segmentation (single-chunk utilities)

Main entry:
    segments_from_chunk(
        clip: cvclip.CVClip,
        seg_processor,
        seg_model,
        max_width: int = 640,
        min_pixels: int = 200,
        keep_labels: list[str] | None = None,
        rle_order: str = "C",
    ) -> tuple[list[dict], tuple[int, int]]

What it does:
    - Grabs the middle frame of ONE chunk (CVClip)
    - Runs SegFormer to get a class map
    - Splits into connected-component instances (per class)
    - Projects to a compact schema:
        [{"seg_id", "label", "bbox", "centroid", "area"}, ...], sorted by area desc
    - Returns that list plus the image size (W, H)

Inputs:
    clip          : a single CVClip (one chunk)
    seg_processor : HF AutoImageProcessor from load_segformer(...)
    seg_model     : HF AutoModelForSemanticSegmentation from load_segformer(...)
    max_width     : optional resize for the frame before inference
    min_pixels    : min component area to keep
    keep_labels   : optional whitelist to filter labels (strings)
    rle_order     : run-length encoding order for stored masks in segments (kept in internal dicts)

Output:
    (segments, (W, H)):
        segments: list[dict] with keys seg_id, label, bbox [x1,y1,x2,y2], centroid [cx,cy], area
        (W, H): image width and height in pixels

Usage:
    from hf_semantic_segmentation import load_segformer, segments_from_chunk
    processor, model = load_segformer("nvidia/segformer-b0-finetuned-ade-512-512")
    segs, (W,H) = segments_from_chunk(clip, processor, model)
"""

from typing import List, Dict, Any, Tuple
import io
import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
# Optional backends for connected components
try:
    import cv2  # noqa
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import scipy.ndimage as ndi  # noqa
    _HAS_NDI = True
except Exception:
    _HAS_NDI = False

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# Use our OpenCV-based clip shim
from cvclip import CVClip


# ------------------------------ Device ------------------------------

def _get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device = _get_device()


# ------------------------ Basic image helpers ------------------------

def load_image(url_or_path: str) -> Image.Image:
    """
    Load an image from an http(s) URL or local path and return RGB PIL.Image.
    """
    if url_or_path.startswith(("http://", "https://")):
        resp = requests.get(url_or_path, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(url_or_path).convert("RGB")


def show_mask_overlay(image: Image.Image, mask_bool: np.ndarray, title: str = "Mask overlay", alpha: float = 0.4):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(mask_bool, alpha=alpha)
    plt.axis("off")
    plt.title(title)
    plt.show()


def middle_frame_pil(clip: CVClip, max_width=640):
    t = float(getattr(clip, "duration", 0.0) or 0.0) / 2.0
    frame = clip.get_frame(t if t > 0 else 0.0)
    img = Image.fromarray(frame)
    if max_width and img.width > max_width:
        h = int(img.height * (max_width / float(img.width)))
        img = img.resize((max_width, h), Image.BILINEAR)
    return img


# ------------------------- SegFormer loading -------------------------

def load_segformer(model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return processor, model


# ---------------------- Segmentation + instances ----------------------

@torch.no_grad()
def segformer_predict_labels(
    image: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
) -> Tuple[np.ndarray, Dict[int, str], List[str]]:
    """
    Run semantic segmentation and return:
      seg_pred: 2D numpy array [H, W] of class IDs
      id2label: dict[int, str]
      labels_in_image: sorted list of unique label names present
    """
    seg_inputs = processor(images=image, return_tensors="pt").to(device)
    logits = model(**seg_inputs).logits  # [1, C, h, w]

    # Upsample logits to original image size (H, W)
    upsampled = F.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False
    )
    seg_pred = upsampled.argmax(dim=1)[0].detach().cpu().numpy()

    id2label = model.config.id2label
    uniq = np.unique(seg_pred).tolist()
    labels_in_image = sorted({id2label[i] for i in uniq if i in id2label})
    return seg_pred, id2label, labels_in_image


def _connected_components(binary: np.ndarray) -> np.ndarray:
    """
    8-connected components on a boolean mask.
    Prefers OpenCV; falls back to SciPy; else pure-NumPy BFS (slower).
    """
    assert binary.dtype == np.bool_ or binary.dtype == bool, "binary must be boolean"

    if _HAS_CV2:
        import cv2 as _cv2  # local import
        num, labels = _cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)

    if _HAS_NDI:
        import scipy.ndimage as _ndi  # local import
        labels, num = _ndi.label(binary)
        return labels.astype(np.int32)

    # ---- Pure NumPy BFS fallback ----
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros_like(binary, dtype=bool)
    current = 0
    from collections import deque
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(H):
        for x in range(W):
            if binary[y, x] and not visited[y, x]:
                current += 1
                q = deque([(y, x)])
                visited[y, x] = True
                labels[y, x] = current
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = current
                            q.append((ny, nx))
    return labels


# --- JSON-friendly RLE for boolean masks ---
def rle_encode_bool(mask: np.ndarray, order: str = "C") -> Dict[str, Any]:
    """
    Simple run-length encoding for a boolean mask for JSON storage.
    Returns:
      {"size": [H, W], "counts": [int,...], "order": "C"|"F", "first_val": 0|1}
    """
    assert mask.dtype == np.bool_ or mask.dtype == bool, "mask must be boolean"
    H, W = mask.shape
    flat = mask.flatten(order=order)
    first_val = int(flat[0])
    counts = []
    run_val = first_val
    run_len = 1
    for v in flat[1:]:
        v = int(v)
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = v
            run_len = 1
    counts.append(run_len)
    return {"size": [int(H), int(W)], "counts": counts, "order": order, "first_val": first_val}


def _bbox_centroid_from_mask(mask_bool: np.ndarray) -> Tuple[List[int], Tuple[float, float], int]:
    ys, xs = np.where(mask_bool)
    area = int(xs.size)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    return [x1, y1, x2, y2], (cx, cy), area

def build_instance_blobs(
    seg_pred: np.ndarray,
    id2label: Dict[int, str],
    min_pixels: int = 50,
    keep_labels: Optional[List[str]] = None,
    rle_order: str = "C",
) -> List[Dict[str, Any]]:
    """
    Split the semantic map (HxW class IDs) into instance blobs per class (connected components).
    Returns list of dicts:
      {"seg_id": int, "label": str, "class_id": int,
       "mask_rle": {...}, "bbox": [x1,y1,x2,y2], "centroid": [cx,cy], "area": int}
    """
    segments: List[Dict[str, Any]] = []
    seg_id = 1

    for cid in [int(i) for i in np.unique(seg_pred)]:
        label = id2label.get(cid, f"class_{cid}")
        if keep_labels and label not in keep_labels:
            continue

        binary = (seg_pred == cid)
        if not binary.any():
            continue

        comp = _connected_components(binary)  # 0 background
        n_comp = int(comp.max())
        if n_comp == 0:
            continue

        for k in range(1, n_comp + 1):
            mask_k = (comp == k)
            area = int(mask_k.sum())
            if area < min_pixels:
                continue

            bbox, (cx, cy), _ = _bbox_centroid_from_mask(mask_k)
            segments.append({
                "seg_id": seg_id,
                "label": label,
                "class_id": cid,
                "mask_rle": rle_encode_bool(mask_k, order=rle_order),
                "bbox": bbox,
                "centroid": [cx, cy],  # (x,y) pixel coords
                "area": area,
            })
            seg_id += 1

    segments.sort(key=lambda d: d["area"], reverse=True)
    return segments


# ------------------- Public API: single-chunk inference -------------------
def segments_from_chunk(
    clip: CVClip,
    seg_processor,
    seg_model,
    max_width: int = 640,
    min_pixels: int = 200,
    keep_labels: Optional[List[str]] = None,
    rle_order: str = "C",
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Run semantic segmentation on the middle frame of ONE chunk (CVClip),
    split into instance blobs, and return a compact list of segments plus (W, H).

    Returns:
        (segments, (W, H))
    """
    image = middle_frame_pil(clip, max_width=max_width)
    seg_pred, id2label, _ = segformer_predict_labels(
        image=image, processor=seg_processor, model=seg_model
    )
    segments = build_instance_blobs(
        seg_pred=seg_pred,
        id2label=id2label,
        min_pixels=min_pixels,
        keep_labels=keep_labels,
        rle_order=rle_order,
    )
    # project to compact schema
    simple = [
        {
            "seg_id": s["seg_id"],
            "label": s["label"],
            "bbox": s["bbox"],  # [x1, y1, x2, y2]
            "centroid": [round(float(s["centroid"][0]), 1), round(float(s["centroid"][1]), 1)],
            "area": s["area"],
        }
        for s in segments
    ]
    return simple, (image.width, image.height)


# --------------- (Optional) SAM helpers kept as provided ---------------

from transformers import SamProcessor, SamModel

def load_sam(model_id: str = "facebook/sam-vit-huge"):
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def sam_best_mask_from_point(
    image: Image.Image,
    sam_processor: SamProcessor,
    sam_model: SamModel,
    point_xy: Tuple[int, int],
    multimask_output: bool = True,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    input_points = [[[int(point_xy[0]), int(point_xy[1])]]]
    sam_inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to(device)
    outputs = sam_model(**sam_inputs, multimask_output=multimask_output)

    post_masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu()
    )
    iou = outputs.iou_scores.detach().cpu()
    if iou.ndim == 3:
        iou_vec = iou[0, 0]
    elif iou.ndim == 2:
        iou_vec = iou[0]
    else:
        iou_vec = iou.view(-1)

    num_masks_pred = int(iou_vec.numel())
    best_idx = int(torch.argmax(iou_vec).item())

    pm = post_masks[0]
    pm_t = pm if isinstance(pm, torch.Tensor) else torch.as_tensor(pm)
    if pm_t.ndim == 4:
        best_idx = min(best_idx, pm_t.shape[1]-1)
        mask = pm_t[0, best_idx]
    elif pm_t.ndim == 3:
        best_idx = min(best_idx, pm_t.shape[0]-1)
        mask = pm_t[best_idx]
    else:
        raise ValueError(f"Unexpected post_masks shape: {tuple(pm_t.shape)}")

    mask_np = mask.detach().cpu().numpy()
    mask_bool = mask_np > threshold
    return mask_bool, {"best_idx": best_idx, "num_masks_pred": num_masks_pred}


def labels_inside_mask(seg_pred: np.ndarray, id2label: Dict[int, str], mask_bool: np.ndarray) -> List[str]:
    ids = np.unique(seg_pred[mask_bool]).tolist()
    return sorted({id2label[i] for i in ids if i in id2label})


__all__ = [
    "load_segformer",
    "segments_from_chunk",
    "segformer_predict_labels",
    "build_instance_blobs",
    "rle_encode_bool",
    "labels_inside_mask",
    "middle_frame_pil",
]
