#!/usr/bin/env python3
"""
Render per-chunk graphs from the pipeline_result JSON produced by main.py.

Usage:
  python draw_chunk_graphs.py --input pipeline_result.json --outdir graphs

What it does:
- For each chunk_k in pipeline_result["ChunkGraphs"]:
    * Adds nodes = FinalLabels
    * Adds solid edges for each (label1, label2, description) in Interactions
- For the LAST chunk only:
    * Parses pipeline_result["Predictions"]["events"] to infer (label_a, label_b, action)
    * Overlays those as DOTTED edges on the same node set

Notes:
- Edge labels are trimmed for readability.
- If a prediction mentions only one known label, it will connect that label to a
  pseudo-node "next_event" (created once) using a dotted edge.

Prereqs:
  pip install networkx matplotlib

"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Set, Optional

import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------- I/O helpers -----------------------------------

def load_pipeline_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------- Pred parsing helpers ----------------------------

def _words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", s.lower())


def _find_label_pairs_in_text(text: str, known_labels: Set[str]) -> List[Tuple[str, str]]:
    """
    Heuristic: return all distinct (a,b) with a != b where BOTH appear as full tokens in text.
    Keeps order of first appearance to prefer human-readable labeling.
    """
    toks = _words(text)
    seen = []
    for i, t in enumerate(toks):
        if t in known_labels:
            for j in range(i + 1, len(toks)):
                u = toks[j]
                if u in known_labels and u != t:
                    pair = (t, u)
                    if pair not in seen:
                        seen.append(pair)
    return seen


def _extract_action_phrase(text: str, a: str, b: str) -> str:
    """
    Try to derive a compact action/verb phrase between labels a and b.
    Strategy:
      - Remove 'a'/'b' tokens and filler words; trim to ~3 words.
    Fallback: the whole text (trimmed).
    """
    t = " " + text.lower() + " "
    # crude boundary padding to help whole-word replaces
    for lab in sorted({a, b}, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(lab)}\b", " ", t)
    # common filler
    t = re.sub(r"\b(the|a|an|to|is|are|continues|keep|keeps|finish(es)?|will|likely|may|might|of|in|on|at|with|closer)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # keep a short phrase
    words = _words(t)
    return " ".join(words[:3]) if words else "predicted"


def predictions_to_edges(events: List[str], known_labels: Set[str]) -> List[Tuple[str, str, str]]:
    """
    Turn prediction strings into dotted-edge tuples (a, b, action).
    Heuristics:
      - If two known labels appear: connect them, label by action phrase.
      - If only one known label appears: connect label -> 'next_event' pseudo-node.
      - If none appear: skip.
    """
    faux = "next_event"
    out = []
    for ev in events or []:
        pairs = _find_label_pairs_in_text(ev, known_labels)
        if pairs:
            # Use first plausible pair and derive short action phrase
            a, b = pairs[0]
            act = _extract_action_phrase(ev, a, b)
            out.append((a, b, act or "predicted"))
        else:
            toks = _words(ev)
            hits = [lab for lab in known_labels if lab in toks]
            if len(hits) >= 1:
                a = hits[0]
                act = _extract_action_phrase(ev, a, a)
                out.append((a, faux, act or "predicted"))
            # else: skip line (no tie to graph)
    return out


# ------------------------------ Drawing core ---------------------------------

def draw_chunk_graph(
    chunk_key: str,
    nodes: List[str],
    solid_edges: List[Tuple[str, str, str]],
    dotted_edges: Optional[List[Tuple[str, str, str]]] = None,
    outdir: str = ".",
    seed: int = 7,
) -> str:
    """
    Draw a single chunk graph.
      nodes: list of labels (lowercase)
      solid_edges: [(u, v, label)]
      dotted_edges: [(u, v, label)] for predictions (last chunk)
    Returns saved PNG path.
    """
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    # Add pseudo-node if dotted edges reference it
    if dotted_edges:
        for (u, v, _) in dotted_edges:
            if u not in G:
                G.add_node(u)
            if v not in G:
                G.add_node(v)

    # Layout once
    pos = nx.spring_layout(G, seed=seed, k=1.1)

    # Base nodes
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=900, linewidths=1.2, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=9)

    # Solid edges (observed interactions)
    if solid_edges:
        e_solid = [(u, v) for (u, v, _) in solid_edges]
        nx.draw_networkx_edges(G, pos, edgelist=e_solid, width=2.0)
        # Edge labels (truncate)
        labels_solid = {(u, v): (lbl if len(lbl) <= 28 else lbl[:25] + "…") for (u, v, lbl) in solid_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_solid, font_size=8)

    # Dotted edges (predictions)
    if dotted_edges:
        e_dotted = [(u, v) for (u, v, _) in dotted_edges]
        nx.draw_networkx_edges(G, pos, edgelist=e_dotted, width=2.0, style="dashed", alpha=0.8)
        labels_dotted = {(u, v): (lbl if len(lbl) <= 28 else lbl[:25] + "…") for (u, v, lbl) in dotted_edges}
        # Nudge dotted labels slightly by reusing edge_labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_dotted, font_size=8, rotate=False)

    plt.axis("off")
    title = f"{chunk_key} (dotted = predicted)" if dotted_edges else f"{chunk_key}"
    plt.title(title)

    ensure_outdir(outdir)
    out_path = os.path.join(outdir, f"{chunk_key}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ------------------------------ Orchestration -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to pipeline_result.json (dumped from main.py)")
    ap.add_argument("--outdir", default="graphs", help="Output folder for PNGs")
    args = ap.parse_args()

    data = load_pipeline_result(args.input)

    # Expect these keys as produced by main.py
    chunk_graphs: Dict[str, Dict[str, any]] = data.get("ChunkGraphs", {})
    if not chunk_graphs:
        raise SystemExit("No 'ChunkGraphs' found in input JSON. Make sure you dumped pipeline_result from main.py")

    # Determine last chunk key in numeric order (chunk_1, chunk_2, ...)
    def chunk_num(k: str) -> int:
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 0

    ordered_keys = sorted(chunk_graphs.keys(), key=chunk_num)
    last_key = ordered_keys[-1]

    # Build prediction dotted edges for last chunk (if available)
    predictions = data.get("Predictions") or {}
    pred_events: List[str] = predictions.get("events") or []
    current_objs: List[str] = predictions.get("current_objects") or []

    # Normalize labels/nodes once
    current_nodes_set = {s.lower().strip() for s in (chunk_graphs[last_key].get("FinalLabels") or [])}
    # Some pipelines include both "FinalLabels" and "current_objects"—merge to be safe
    if current_objs:
        current_nodes_set |= {s.lower().strip() for s in current_objs}

    dotted_edges_last = predictions_to_edges(pred_events, current_nodes_set) if pred_events else []

    # Render each chunk
    print(f"[draw] Found {len(ordered_keys)} chunks. Writing PNGs to: {args.outdir}")
    for k in ordered_keys:
        payload = chunk_graphs[k]
        nodes = [str(x).lower() for x in (payload.get("FinalLabels") or [])]

        solid_edges = []
        for tup in (payload.get("Interactions") or []):
            # Expected shape: (l1, l2, description)
            if isinstance(tup, (list, tuple)) and len(tup) >= 3:
                l1 = str(tup[0]).lower().strip()
                l2 = str(tup[1]).lower().strip()
                desc = str(tup[2]).strip()
                solid_edges.append((l1, l2, desc))

        dotted = dotted_edges_last if k == last_key else None
        out_png = draw_chunk_graph(k, nodes, solid_edges, dotted_edges=dotted, outdir=args.outdir)
        print(f"[draw] Wrote {out_png}")

    print("[draw] Done.")


if __name__ == "__main__":
    main()
