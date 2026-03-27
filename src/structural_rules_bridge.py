"""
Structural Rules Bridge
=======================
Connects steel_mesh_structural_rules_1.py and fem_solver.py to the pipeline.

Responsibilities:
  1. compute_mesh_descriptors(ge)  → compact dict for Gemini prompt + optimizer
  2. run_code_checks(fea_results, mesh_desc) → IBC 2024 / ASCE 7-22 check report
  3. section_upgrade_suggestion(check_report, graph) → list of member upgrade hints

Note on coordinates: Blender OBJ is Y-up (Y = height, Z = depth).
steel_mesh_structural_rules_1 uses Z-up convention (Z = height).
This bridge swaps Y↔Z before calling the rules functions, and swaps back
for any results that reference coordinates.
"""
from __future__ import annotations
import sys, os
import math
import numpy as np

# Ensure repo root is on path so we can import the rules module
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Section mass table (kg/m) for carbon/cost metrics ────────────────────────
SECTION_MASS_KG_PER_M = {
    "W8x31":               46.2,
    "IPE_300":             42.2,
    "HEA_200":             42.3,
    "HSS6x0.500":          37.0,
    "W12x50":              74.4,
    "W12x53":              78.9,
    "Tubular_HSS_4x4x1/4": 25.0,
    "HSS8x8x0.500":        76.9,
    "HSS10x0.500":         48.0,
    "W14x90":             133.9,
    "W18x97":             144.3,
    "W24x146":            217.3,
    "W14x159":            236.7,
    "HSS12x12x0.625":     121.0,
    "HSS16x0.625":         61.8,
    "W14x283":            421.4,
}

# ── Per-section radius-of-gyration (minor axis, inches) for slenderness ──────
SECTION_RY_IN = {
    "W8x31": 1.27, "IPE_300": 1.43, "HEA_200": 2.09,
    "HSS6x0.500": 1.97, "HSS8x8x0.500": 3.04, "HSS10x0.500": 3.44,
    "W12x50": 1.96, "W12x53": 2.48, "W14x90": 3.70,
    "W14x159": 4.00, "W14x283": 4.17, "W18x97": 2.65,
    "W24x146": 2.97, "HSS12x12x0.625": 4.59, "HSS16x0.625": 5.34,
    "Tubular_HSS_4x4x1/4": 1.52, "HSS6x0.500": 1.97,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1 — MESH DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_mesh_descriptors(ge) -> dict:
    """
    Extract structural geometry descriptors from a GeometryEngine instance.
    Returns a compact, JSON-serializable dict used by the AI prompt and optimizer.
    """
    try:
        from steel_mesh_structural_rules_1 import (
            extract_mesh_descriptors, select_structural_system,
        )

        verts_orig = np.array(ge.mesh.vertices, dtype=float)
        faces      = np.array(ge.mesh.faces, dtype=int)

        # Swap Y↔Z so rules module sees Z as vertical
        verts_zup = verts_orig[:, [0, 2, 1]]  # (X, Z_blender, Y_blender) → (X, Y_rules, Z_rules=height)

        creases_data = ge.extract_primary_creases(angle_threshold_degrees=5.0)
        if creases_data["edges"]:
            creases = np.array(creases_data["edges"], dtype=int)
        else:
            creases = np.zeros((0, 2), dtype=int)

        desc = extract_mesh_descriptors(verts_zup, faces, creases, floor_height_m=4.0)
        sys_sel = select_structural_system(desc, sdc="D", program="office")

        return {
            "height_m":         round(desc.height_m, 2),
            "width_m":          round(desc.width_m, 2),
            "depth_m":          round(desc.depth_m, 2),
            "H_W_ratio":        round(desc.H_W_ratio, 3),
            "aspect_category":  desc.aspect_category,
            "curvature_type":   desc.curvature_type,
            "story_count":      desc.story_count,
            "stories_per_module": desc.stories_per_module,
            "void_fraction":    round(desc.void_fraction, 3),
            "lateral_system":   sys_sel.lateral_system,
            "gravity_system":   sys_sel.gravity_system,
            "has_mega_truss":   sys_sel.has_mega_truss,
            "has_outriggers":   sys_sel.has_outriggers,
            "n_outrigger_levels": sys_sel.n_outrigger_levels,
            "seismic_R":        sys_sel.seismic_system.get("R", 8.0),
            "seismic_Cd":       sys_sel.seismic_system.get("Cd", 5.5),
            "material":         sys_sel.material,
            "notes":            sys_sel.notes,
        }

    except Exception as e:
        import traceback
        print(f"[rules_bridge] compute_mesh_descriptors error: {e}")
        traceback.print_exc()
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 2 — CODE CHECKS (IBC 2024 / ASCE 7-22 / AISC 360-22)
# ══════════════════════════════════════════════════════════════════════════════

def run_code_checks(fea_results: dict, mesh_desc: dict, graph=None) -> dict:
    """
    Run IBC 2024 / ASCE 7-22 / AISC 360-22 code checks on FEA results.

    Returns
    -------
    {
      "overall": "pass" | "warn" | "fail",
      "checks": [...],
      "max_disp_m": float,
      "drift_DCR": float,
      "failing_members": [{"member": str, "reason": str}]
    }
    """
    try:
        from steel_mesh_structural_rules_1 import (
            check_story_drift, check_beam_deflection, check_slenderness,
            DRIFT_SEISMIC_SDC_DF, DEFLECTION_LL_LIMIT, SLENDERNESS_MAX,
        )
    except ImportError as e:
        print(f"[rules_bridge] Cannot import rules module: {e}")
        return {"overall": "unknown", "checks": [], "max_disp_m": 0.0, "drift_DCR": 0.0, "failing_members": []}

    checks = []
    max_disp = fea_results.get("max_displacement", 0.0)
    height_m = mesh_desc.get("height_m", 10.0) or 10.0
    story_count = max(1, mesh_desc.get("story_count", 3))
    h_sx = height_m / story_count  # average story height (m)

    # ── 1. Global seismic drift check (ASCE 7-22 §12.12-1) ──────────────────
    Cd   = mesh_desc.get("seismic_Cd", 5.5)
    Ie   = 1.0
    # Amplified displacement = max_disp * Cd / Ie  (raw FEA gives elastic δ_xe)
    delta_amplified = max_disp * Cd / Ie
    drift_check = check_story_drift("GLOBAL", delta_m=delta_amplified, h_sx_m=h_sx, sdc="D")
    checks.append({
        "id": drift_check.element_id,
        "type": drift_check.check_type,
        "DCR": round(drift_check.DCR, 3),
        "status": drift_check.status,
        "message": drift_check.message,
    })
    drift_DCR = drift_check.DCR

    # ── 2. Slenderness checks (AISC 360-22 §E2) from graph edges ─────────────
    if graph is not None:
        for node_id_u, node_id_v, edata in graph.edges(data=True):
            try:
                p1 = np.array(graph.nodes[node_id_u]["coords"])
                p2 = np.array(graph.nodes[node_id_v]["coords"])
                L_m = float(np.linalg.norm(p1 - p2))
                sec  = edata.get("section", "W14x90")
                ry   = SECTION_RY_IN.get(sec, 1.5) * 0.0254   # in → m
                K    = 1.0  # assume pinned-pinned (conservative)
                KL_r = K * L_m / max(ry, 0.001)
                if KL_r > SLENDERNESS_MAX:
                    chk = check_slenderness(f"{node_id_u}-{node_id_v}", KL_r)
                    checks.append({
                        "id": chk.element_id, "type": chk.check_type,
                        "DCR": round(chk.DCR, 3), "status": chk.status, "message": chk.message,
                    })
            except Exception:
                pass

    # ── 3. Member deflection failures from PyNite ────────────────────────────
    fea_failures = fea_results.get("failures", [])

    # ── Summarise ─────────────────────────────────────────────────────────────
    all_statuses = [c["status"] for c in checks]
    if "fail" in all_statuses:
        overall = "fail"
    elif "warn" in all_statuses:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "overall": overall,
        "checks": checks,
        "max_disp_m": round(max_disp, 5),
        "drift_DCR": round(drift_DCR, 3),
        "h_sx_m": round(h_sx, 2),
        "failing_members": fea_failures,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3 — SECTION UPGRADE HINTS
# ══════════════════════════════════════════════════════════════════════════════

# Ordered ladders for section upgrades
_CREASE_LADDER = [
    "IPE_300", "W8x31", "W12x50", "W12x53",
    "W14x90", "W18x97", "W24x146", "W14x159", "W14x283"
]
_BRACE_LADDER = [
    "Tubular_HSS_4x4x1/4", "HEA_200", "HSS6x0.500",
    "HSS8x8x0.500", "HSS10x0.500", "HSS12x12x0.625", "HSS16x0.625"
]
_ALL_LADDERS = {s: i for i, s in enumerate(_CREASE_LADDER)}
_ALL_LADDERS.update({s: i for i, s in enumerate(_BRACE_LADDER)})


def section_upgrade_hints(check_report: dict, graph, steps: int = 1) -> list[dict]:
    """
    For members that have slenderness failures, suggest the next section up the ladder.
    Returns a list of {source, target, current_section, suggested_section}.
    """
    hints = []
    if graph is None:
        return hints

    for node_id_u, node_id_v, edata in graph.edges(data=True):
        sec  = edata.get("section", "")
        role = edata.get("section_type", "secondary_lattice")

        # Pick appropriate ladder
        if role == "primary_crease":
            ladder = _CREASE_LADDER
        else:
            ladder = _BRACE_LADDER

        if sec in ladder:
            idx = ladder.index(sec)
            new_idx = min(len(ladder) - 1, idx + steps)
            if new_idx != idx:
                hints.append({
                    "source": node_id_u, "target": node_id_v,
                    "current": sec, "suggested": ladder[new_idx]
                })
    return hints


# ══════════════════════════════════════════════════════════════════════════════
# 4 — RULES DIGEST FOR GEMINI PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def rules_digest_for_prompt(mesh_desc: dict) -> str:
    """
    Produce a compact, structured text digest of the rules that are most
    relevant to this specific mesh, suitable for inclusion in the Gemini prompt.
    """
    if not mesh_desc:
        return ""

    lines = [
        "=== STRUCTURAL RULES DIGEST (auto-computed from mesh) ===",
        "",
        f"BUILDING GEOMETRY:",
        f"  Height = {mesh_desc.get('height_m', '?')} m | Width = {mesh_desc.get('width_m', '?')} m | Depth = {mesh_desc.get('depth_m', '?')} m",
        f"  H/W ratio = {mesh_desc.get('H_W_ratio', '?')} → Category: {mesh_desc.get('aspect_category', '?')}",
        f"  Curvature: {mesh_desc.get('curvature_type', '?')} | Est. stories: {mesh_desc.get('story_count', '?')}",
        f"  Void fraction: {mesh_desc.get('void_fraction', 0):.1%}",
        "",
        f"SELECTED STRUCTURAL SYSTEM (Chapter 19 decision tree):",
        f"  Lateral: {mesh_desc.get('lateral_system', '?')}",
        f"  Gravity: {mesh_desc.get('gravity_system', '?')}",
        f"  Mega-truss needed: {mesh_desc.get('has_mega_truss', False)}",
        f"  Outriggers: {mesh_desc.get('has_outriggers', False)} ({mesh_desc.get('n_outrigger_levels', 0)} levels)",
        f"  Seismic R = {mesh_desc.get('seismic_R', 8)}, Cd = {mesh_desc.get('seismic_Cd', 5.5)}",
        "",
        "MANDATORY CODE LIMITS (must not be exceeded):",
        "  Seismic drift (ASCE 7-22 §12.12-1): Δ·Cd/h_sx ≤ 0.020",
        "  Beam deflection (IBC Table 1604.3): δ_LL ≤ L/360",
        "  Column slenderness (AISC 360 §E2): KL/r ≤ 200",
        "  Member interaction (AISC 360 §H1): Pr/Pc + (8/9)(Mrx/Mcx) ≤ 1.0",
        "",
        "SECTION ASSIGNMENT RULES (Chapter 11 + Appendix B):",
    ]

    H_W = mesh_desc.get("H_W_ratio", 2.0)
    lat = mesh_desc.get("lateral_system", "SMF")

    if H_W > 8:
        lines += [
            "  → SUPER-TALL: Use W14x283 heavy columns + HSS16x0.625 perimeter diagonals.",
            "  → Place outrigger belt trusses at mechanical floors.",
            "  → Ring beams (W24x146) at every diagrid module level.",
        ]
    elif H_W >= 4:
        lines += [
            "  → TALL (diagrid preferred): Use W14x159 / W14x283 at corners,",
            "    W18x97 or W24x146 for long horizontal spans,",
            "    HSS12x12x0.625 or HSS10x0.500 for diagonal braces.",
            "  → Diagrid angle = {:.0f}°.".format(70.0 if H_W < 3 else 65.0 if H_W <= 6 else 60.0),
        ]
    elif H_W >= 1:
        lines += [
            "  → MID-RISE: W14x90 columns, W12x53 medium beams, HSS8x8x0.500 braces.",
            "  → Add X-bracing in all bays > 4 m wide.",
        ]
    else:
        lines += [
            "  → LOW/WIDE (space frame): W12x50 or IPE_300 grid members,",
            "    HSS6x0.500 diagonals, perimeter ring beam W18x97.",
        ]

    if mesh_desc.get("has_mega_truss"):
        lines.append("  → MEGA-TRUSS required at void level: use W14x283 chords + W18x97 verticals.")

    notes = mesh_desc.get("notes", [])
    if notes:
        lines += ["", "SYSTEM SELECTION NOTES:"] + [f"  · {n}" for n in notes]

    lines.append("=== END OF RULES DIGEST ===")
    return "\n".join(lines)
