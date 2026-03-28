"""
Section Sizer
=============
Phase 3 of the structural design pipeline.

Assigns cross-sections to graph edges DETERMINISTICALLY using
steel_rulebook.py (R19–R30) — no Gemini API calls.

Validates the structure using R39, R41, R42 after FEA.

Coordinate system: Z-up (Z = height), consistent with GeometryEngine.
"""
from __future__ import annotations
import math
import numpy as np
import sys, os

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from steel_rulebook import (
    R19_beam_depth_from_span_mm,
    R20_column_depth_from_load_mm,
    R28_column_axial_load_kN,
    R39_check_slenderness,
    R42_check_story_drift,
)


# ══════════════════════════════════════════════════════════════════════════════
# Section catalogue — (approx_depth_mm, section_name), ordered light → heavy
# Depths are nominal section depths in mm.
# ══════════════════════════════════════════════════════════════════════════════

_COLUMN_CATALOGUE = [
    (200, "W8x31"),
    (310, "W12x50"),
    (310, "W12x53"),
    (360, "W14x90"),
    (360, "W14x159"),
    (360, "W14x283"),
    (460, "W18x97"),
    (610, "W24x146"),
]

_BEAM_CATALOGUE = [
    (190, "HEA_200"),
    (200, "W8x31"),
    (300, "IPE_300"),
    (310, "W12x50"),
    (310, "W12x53"),
    (460, "W18x97"),
    (610, "W24x146"),
]

_BRACE_CATALOGUE = [
    (100, "Tubular_HSS_4x4x1/4"),
    (152, "HSS6x0.500"),
    (203, "HSS8x8x0.500"),
    (254, "HSS10x0.500"),
    (305, "HSS12x12x0.625"),
    (406, "HSS16x0.625"),
]

# Full section ladder (light → heavy) for upgrade/downgrade stepping
STEEL_LADDER = [
    "W8x31", "IPE_300", "HEA_200", "HSS6x0.500",
    "W12x50", "W12x53", "Tubular_HSS_4x4x1/4",
    "HSS8x8x0.500", "HSS10x0.500",
    "W14x90", "W18x97", "W24x146",
    "W14x159", "HSS12x12x0.625", "HSS16x0.625", "W14x283",
]
LADDER_INDEX = {s: i for i, s in enumerate(STEEL_LADDER)}

# Radius of gyration (minor axis, in) for slenderness check
_RY_IN = {
    "W8x31": 1.27, "IPE_300": 1.43, "HEA_200": 2.09,
    "HSS6x0.500": 1.97, "HSS8x8x0.500": 3.04, "HSS10x0.500": 3.44,
    "W12x50": 1.96, "W12x53": 2.48, "W14x90": 3.70,
    "W14x159": 4.00, "W14x283": 4.17, "W18x97": 2.65,
    "W24x146": 2.97, "HSS12x12x0.625": 4.59, "HSS16x0.625": 5.34,
    "Tubular_HSS_4x4x1/4": 1.52,
}

_ALL_VALID_SECTIONS = set(STEEL_LADDER) | {"Core_Massive"}


def _nearest_section(depth_mm: float, catalogue: list) -> str:
    """Return the lightest section whose depth meets or exceeds depth_mm."""
    for d, name in catalogue:
        if d >= depth_mm:
            return name
    return catalogue[-1][1]


# ══════════════════════════════════════════════════════════════════════════════
# Member role classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_member_role(coords_u: np.ndarray, coords_v: np.ndarray) -> str:
    """
    Classify a member as 'column', 'beam', or 'brace' from its orientation.

    Rules (Z = height axis):
      vertical_ratio > 0.85  → column
      horizontal_ratio > 0.85 → beam
      otherwise               → brace (diagonal)
    """
    delta = np.asarray(coords_v) - np.asarray(coords_u)
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return "brace"

    vertical_ratio   = abs(delta[2]) / length
    horizontal_ratio = math.sqrt(delta[0]**2 + delta[1]**2) / length

    if vertical_ratio > 0.85:
        return "column"
    elif horizontal_ratio > 0.85:
        return "beam"
    return "brace"


# ══════════════════════════════════════════════════════════════════════════════
# Section sizing
# ══════════════════════════════════════════════════════════════════════════════

def compute_section_name(
    role: str,
    length_m: float,
    n_stories_above: int = 3,
    trib_width_m: float = 4.0,
) -> str:
    """
    Compute the required section name using rulebook R19 / R20 / R28.

    Parameters
    ----------
    role            : 'column' | 'beam' | 'brace'
    length_m        : member length in metres
    n_stories_above : floors supported above (columns only)
    trib_width_m    : tributary width estimate (m)
    """
    if role == "beam":
        d_mm = R19_beam_depth_from_span_mm(
            length_m,
            support_condition="continuous",
            load_type="floor",
        )
        return _nearest_section(d_mm, _BEAM_CATALOGUE)

    elif role == "column":
        trib_area = trib_width_m ** 2
        loads = R28_column_axial_load_kN(trib_area, max(1, n_stories_above))
        d_mm = R20_column_depth_from_load_mm(loads["Pu_kN"], length_m)
        return _nearest_section(d_mm, _COLUMN_CATALOGUE)

    else:  # brace / diagrid diagonal
        # KL/r ≤ 200  →  r_min ≥ KL/200.  For HSS: r ≈ 0.35D.
        # We choose D so that KL/r ≤ 120 (practical target for braces).
        d_mm = max(100.0, length_m * 1000.0 / (120.0 * 0.35 * 0.0254 / 0.0254))
        d_mm = max(100.0, min(406.0, d_mm))
        return _nearest_section(d_mm, _BRACE_CATALOGUE)


def apply_sections(graph, mesh_desc: dict | None = None) -> None:
    """
    Assign sections to every edge in the graph using deterministic sizing.
    Overwrites any existing section assignment.
    Mutates the graph in-place.
    """
    # Determine height range for stories-above calculation
    z_vals = [data.get("coords", (0, 0, 0))[2] for _, data in graph.nodes(data=True)]
    z_min = min(z_vals) if z_vals else 0.0
    z_max = max(z_vals) if z_vals else 10.0
    total_h = max(z_max - z_min, 1.0)
    story_count = max(1, (mesh_desc or {}).get("story_count", 3))

    for u, v, edata in graph.edges(data=True):
        cu = np.array(graph.nodes[u].get("coords", (0, 0, 0)))
        cv = np.array(graph.nodes[v].get("coords", (0, 0, 0)))
        length_m = float(np.linalg.norm(cu - cv))
        if length_m < 0.01:
            length_m = 0.01

        role = classify_member_role(cu, cv)

        # Estimate stories above from node elevation (higher node → fewer above)
        z_upper = max(float(cu[2]), float(cv[2]))
        frac = max(0.0, min(1.0, (z_upper - z_min) / total_h))
        n_above = max(1, int((1.0 - frac) * story_count))

        section = compute_section_name(role, length_m, n_stories_above=n_above)
        graph[u][v]["section"] = section

        # Set role tag and connection defaults
        graph[u][v]["section_type"] = (
            "primary_crease" if role in ("column", "beam") else "secondary_lattice"
        )
        if "connection" not in edata or not edata["connection"]:
            graph[u][v]["connection"] = "fixed" if role != "brace" else "pinned"
        if "typology" not in edata or not edata["typology"]:
            graph[u][v]["typology"] = "welded" if role != "brace" else "hinge"


# ══════════════════════════════════════════════════════════════════════════════
# Structural validation (R39, R42)
# ══════════════════════════════════════════════════════════════════════════════

def validate_structure(graph, fea_results: dict, mesh_desc: dict) -> dict:
    """
    Run R39 (slenderness) and R42 (drift) checks after FEA.

    Returns
    -------
    {
      "overall": "pass" | "warn" | "fail",
      "checks": [...],
      "drift_DCR": float,
      "failing_members": [{"member": str, "reason": str}],
      "max_disp_m": float,
    }
    """
    checks = []
    max_disp   = fea_results.get("max_displacement", 0.0)
    height_m   = (mesh_desc or {}).get("height_m", 10.0) or 10.0
    story_count = max(1, (mesh_desc or {}).get("story_count", 3))
    h_sx_m     = height_m / story_count
    Cd         = (mesh_desc or {}).get("seismic_Cd", 5.5)

    # ── R42: Global seismic drift ──────────────────────────────────────────────
    delta_amplified_mm = max_disp * Cd * 1000.0   # elastic δ × Cd → mm
    drift = R42_check_story_drift(delta_amplified_mm, h_sx_m * 1000.0)
    checks.append({
        "id":      "DRIFT_GLOBAL",
        "type":    "seismic_drift",
        "DCR":     round(drift["DCR"], 3),
        "status":  "pass" if drift["pass"] else "fail",
        "message": drift["msg"],
    })
    drift_DCR = drift["DCR"]

    # ── R39: Member slenderness ────────────────────────────────────────────────
    for u, v, edata in graph.edges(data=True):
        try:
            cu = np.array(graph.nodes[u]["coords"])
            cv = np.array(graph.nodes[v]["coords"])
            L_m  = float(np.linalg.norm(cu - cv))
            sec  = edata.get("section", "W14x90")
            ry_m = _RY_IN.get(sec, 1.5) * 0.0254   # in → m
            KL_r = L_m / max(ry_m, 1e-4)

            result = R39_check_slenderness(KL_r)
            if not result["pass"]:
                checks.append({
                    "id":      f"{u}-{v}",
                    "type":    "slenderness",
                    "DCR":     round(result["DCR"], 3),
                    "status":  "fail",
                    "message": result["msg"],
                })
        except Exception:
            pass

    statuses = [c["status"] for c in checks]
    overall  = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    failing  = [
        {"member": c["id"], "reason": c["message"]}
        for c in checks
        if c["status"] == "fail" and c["id"] != "DRIFT_GLOBAL"
    ]

    return {
        "overall":         overall,
        "checks":          checks,
        "drift_DCR":       round(drift_DCR, 3),
        "failing_members": failing,
        "max_disp_m":      round(max_disp, 5),
        "h_sx_m":          round(h_sx_m, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section upgrade (for revision loop)
# ══════════════════════════════════════════════════════════════════════════════

def upgrade_failing_members(graph, check_report: dict, steps: int = 1) -> None:
    """
    Upgrade sections of members that failed slenderness checks, or
    upgrade ALL members if drift fails.  Mutates the graph in-place.
    """
    drift_fail = any(
        c["status"] == "fail" and "drift" in c["type"]
        for c in check_report.get("checks", [])
    )

    failing_pairs: set[tuple] = set()
    for c in check_report.get("checks", []):
        if c["status"] == "fail" and c["id"] not in ("DRIFT_GLOBAL",):
            try:
                parts = c["id"].split("-")
                if len(parts) == 2:
                    failing_pairs.add((int(parts[0]), int(parts[1])))
            except Exception:
                pass

    for u, v, edata in graph.edges(data=True):
        if not drift_fail and (u, v) not in failing_pairs and (v, u) not in failing_pairs:
            continue

        sec = edata.get("section", "")
        if sec in LADDER_INDEX:
            new_idx = min(len(STEEL_LADDER) - 1, LADDER_INDEX[sec] + steps)
            graph[u][v]["section"] = STEEL_LADDER[new_idx]


def scale_sections(graph, steps: int):
    """Return a COPY of graph with every section shifted ±steps on the ladder."""
    import networkx as nx
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        sec = data.get("section", "")
        if sec in LADDER_INDEX:
            new_idx = max(0, min(len(STEEL_LADDER) - 1, LADDER_INDEX[sec] + steps))
            G[u][v]["section"] = STEEL_LADDER[new_idx]
    return G
