"""
Mesh Processor
==============
Phase 1 of the structural design pipeline.

Extracts candidate structural nodes from the mesh using:
  - Crease edges (dihedral angle analysis) → primary structural backbone
  - Story-level boundary sampling (R08 slicing planes)
  - Spatial filtering to ≤ max_count nodes for Gemini token budget

Coordinate system: Z-up (GeometryEngine normalises Z_min → 0).
"""
from __future__ import annotations
import numpy as np
import sys, os

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Story slicing ──────────────────────────────────────────────────────────────

def get_story_levels(ge, floor_height_m: float = 4.0) -> list[float]:
    """
    Return Z-coordinates of horizontal slicing planes using R08 from steel_rulebook.

    Falls back to simple linear spacing when the rulebook is unavailable.
    """
    try:
        from steel_rulebook import R08_story_slicing_planes_m
        b = ge.mesh.bounds
        z_min, z_max = float(b[0][2]), float(b[1][2])
        return R08_story_slicing_planes_m(z_min, z_max, floor_height_m=floor_height_m)
    except Exception:
        b = ge.mesh.bounds
        z_min, z_max = float(b[0][2]), float(b[1][2])
        n = max(1, int((z_max - z_min) / floor_height_m))
        return [z_min + i * floor_height_m for i in range(n + 1)]


# ── Candidate node extraction ──────────────────────────────────────────────────

def get_candidate_nodes(
    ge,
    max_count: int = 80,
    crease_angle_deg: float = 15.0,
    floor_height_m: float = 4.0,
) -> tuple[list[dict], list[dict]]:
    """
    Build a prioritised list of structural candidate nodes.

    Priority
    --------
    1. Crease nodes (dihedral angle > crease_angle_deg) — structural backbone.
    2. Floor-boundary nodes near each story Z-level.
    3. Spatial subsampling when the total exceeds max_count.

    Returns
    -------
    nodes : list of {"id": int, "x": float, "y": float, "z": float, "tag": str}
            tag ∈ {"ground", "crease", "floor"}
    edges : list of {"source": int, "target": int, "type": "primary_crease"}
            — re-mapped to new sequential IDs
    """
    all_verts = ge.mesh.vertices
    b = ge.mesh.bounds
    z_min = float(b[0][2])
    ground_threshold = z_min + 0.5   # nodes within 0.5 m of base = ground

    # ── 1. Crease nodes ────────────────────────────────────────────────────────
    creases = ge.extract_primary_creases(angle_threshold_degrees=crease_angle_deg)
    crease_idx_set = set(int(i) for i in creases["nodes"])
    raw_crease_edges = creases["edges"]          # list of [u_orig, v_orig]

    orig_to_new: dict[int, int] = {}             # original vertex index → new id
    nodes: list[dict] = []

    def _add_node(orig_idx: int, tag: str):
        if orig_idx in orig_to_new:
            return
        new_id = len(nodes)
        orig_to_new[orig_idx] = new_id
        v = all_verts[orig_idx]
        nodes.append({
            "id":  new_id,
            "x":   round(float(v[0]), 4),
            "y":   round(float(v[1]), 4),
            "z":   round(float(v[2]), 4),
            "tag": "ground" if v[2] <= ground_threshold else tag,
        })

    for idx in sorted(crease_idx_set):
        _add_node(idx, "crease")

    # ── 2. Floor-boundary nodes ────────────────────────────────────────────────
    story_levels = get_story_levels(ge, floor_height_m=floor_height_m)
    for z_level in story_levels[1:]:             # skip ground plane
        dists = np.abs(all_verts[:, 2] - z_level)
        nearby = np.where(dists < min(0.8, floor_height_m * 0.2))[0]
        for idx in nearby:
            _add_node(int(idx), "floor")

    # ── 3. Trim if over budget ─────────────────────────────────────────────────
    if len(nodes) > max_count:
        essential = [n for n in nodes if n["tag"] in ("ground", "crease")]
        non_essential = [n for n in nodes if n["tag"] == "floor"]

        budget = max(0, max_count - len(essential))
        if budget > 0 and non_essential:
            # Spatially subsample: keep every nth
            step = max(1, len(non_essential) // budget)
            kept_non_essential = non_essential[::step][:budget]
        else:
            kept_non_essential = []

        # Rebuild with fresh IDs
        kept = essential + kept_non_essential
        orig_to_new = {}
        nodes = []
        for n in kept:
            # recover original index by reverse lookup — use coords
            new_id = len(nodes)
            orig_to_new[_find_orig(all_verts, n)] = new_id
            n["id"] = new_id
            nodes.append(n)

    # ── 4. Re-map crease edges to new IDs ─────────────────────────────────────
    edges: list[dict] = []
    for e in raw_crease_edges:
        u_orig, v_orig = int(e[0]), int(e[1])
        if u_orig in orig_to_new and v_orig in orig_to_new:
            edges.append({
                "source": orig_to_new[u_orig],
                "target": orig_to_new[v_orig],
                "type": "primary_crease",
            })

    return nodes, edges


def _find_orig(all_verts: np.ndarray, node: dict) -> int:
    """Locate original vertex index by coordinate match (used during trimming)."""
    target = np.array([node["x"], node["y"], node["z"]])
    dists = np.linalg.norm(all_verts - target, axis=1)
    return int(np.argmin(dists))


# ── Node tagging helpers ───────────────────────────────────────────────────────

def tag_ground_nodes(nodes: list[dict]) -> list[int]:
    """Return new-ID list of ground nodes (to be used as pinned supports)."""
    return [n["id"] for n in nodes if n["tag"] == "ground"]


def get_mesh_bounds(ge) -> dict:
    """Return bounding box dimensions as a dict."""
    b = ge.mesh.bounds
    return {
        "x_min":    float(b[0][0]),  "x_max": float(b[1][0]),
        "y_min":    float(b[0][1]),  "y_max": float(b[1][1]),
        "z_min":    float(b[0][2]),  "z_max": float(b[1][2]),
        "height_m": float(b[1][2] - b[0][2]),
        "width_m":  float(b[1][0] - b[0][0]),
        "depth_m":  float(b[1][1] - b[0][1]),
    }
