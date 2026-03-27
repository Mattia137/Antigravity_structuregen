"""
COMPLEX STEEL STRUCTURE GENERATION — MATHEMATICAL RULES & FUNCTIONS
====================================================================
Derived from: Complex Non-Hierarchical Structural Systems AI Design Manual
Codes: IBC 2024 · ASCE 7-22 · AISC 360-22 · ACI 318-19 · AISC 341-22

This module provides:
  1. Mesh geometry descriptors
  2. System selection rules (decision tree)
  3. Member sizing functions
  4. Load / load-combination generators
  5. Code-check constraint functions (pass / warn / fail)
  6. Diagrid / space-frame / core generation algorithms
  7. Numeric domain limits (all labelled constants)

All functions accept plain Python numeric types / numpy arrays so they can
be called from any CAD / mesh pipeline.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — GLOBAL NUMERIC CONSTANTS  (domain limits)
# ══════════════════════════════════════════════════════════════════════════════

# ── Material strengths (ksi) ──────────────────────────────────────────────────
Fy_A992      = 50.0        # W-shapes, ksi  (A992 steel)
Fy_A500C     = 46.0        # HSS round / rect, ksi  (A500 Gr.C)
Fy_A36       = 36.0        # plates / built-up, ksi
Fu_A992      = 65.0        # ultimate tensile, ksi
E_STEEL      = 29_000.0    # Young's modulus, ksi
G_STEEL      = 11_200.0    # shear modulus, ksi

# ── Concrete ─────────────────────────────────────────────────────────────────
fc_CORE      = 6.0         # f'c concrete core, ksi  (6000 psi typical)
Ec_CONCRETE  = 57.0 * math.sqrt(fc_CORE * 1000)  # ksi  (ACI 318 §19.2.2)
DENSITY_CONC = 150.0       # pcf
DENSITY_STEEL= 490.0       # pcf
DENSITY_CLT  = 33.0        # pcf  (mid-range glulam/CLT)

# ── LRFD load factors (ASCE 7-22 §2.3.1) ─────────────────────────────────────
LF = {
    "LC1": {"D": 1.4},
    "LC2": {"D": 1.2, "L": 1.6, "Lr_S_R": 0.5},
    "LC3": {"D": 1.2, "Lr_S_R": 1.6, "L_or_halfW": 1.0},
    "LC4": {"D": 1.2, "W": 1.0, "L": 1.0, "Lr_S_R": 0.5},
    "LC5": {"D": 0.9, "W": 1.0},
    "LC6": {"D": 1.2, "E": 1.0, "L": 1.0, "S": 0.2},
    "LC7": {"D": 0.9, "E": 1.0},
}

# ── Seismic R-factors, overstrength Ω₀, deflection amplification Cd ──────────
SEISMIC_SYSTEMS = {
    "SMF":   {"R": 8.0,  "omega": 3.0,  "Cd": 5.5, "height_limit_ft": math.inf},
    "SCBF":  {"R": 6.0,  "omega": 2.0,  "Cd": 5.0, "height_limit_ft": 160.0},
    "BRBF":  {"R": 8.0,  "omega": 2.5,  "Cd": 5.0, "height_limit_ft": 160.0},
    "EBF":   {"R": 8.0,  "omega": 2.0,  "Cd": 4.0, "height_limit_ft": 160.0},
    "DUAL_SMF_SCBF": {"R": 7.0, "omega": 2.5, "Cd": 5.5, "height_limit_ft": math.inf},
    "RC_SPECIAL_WALL": {"R": 6.0, "omega": 2.5, "Cd": 5.0, "height_limit_ft": math.inf},
    "DIAGRID_SCBF":    {"R": 6.0, "omega": 2.0, "Cd": 5.0, "height_limit_ft": None},
    "DIAGRID_EBF":     {"R": 8.0, "omega": 2.0, "Cd": 4.0, "height_limit_ft": None},
}

# ── Serviceability limits ─────────────────────────────────────────────────────
DRIFT_SEISMIC_SDC_DF   = 0.020      # Δ/h_sx  (ASCE 7-22 Table 12.12-1, office)
DRIFT_WIND_LIMIT       = 1 / 400    # H/400  (AISC DG 3)
DEFLECTION_LL_LIMIT    = 1 / 360    # L/360  (IBC Table 1604.3)
DEFLECTION_TL_LIMIT    = 1 / 240    # L/240
DEFLECTION_ROOF_LL     = 1 / 480    # L/480 long-span roof (span > 25 ft)
FLOOR_VIB_MIN_FREQ_HZ  = 4.0       # Hz  (AISC DG 11)
TCC_VIB_MIN_FREQ_HZ    = 8.0       # Hz  (TCC floor recommendation)
SLENDERNESS_MAX        = 200.0      # KL/r  (AISC 360 §E2, compression)

# ── Composite beam rules (AISC 360 Ch. I) ────────────────────────────────────
COMPOSITE_MIN_RATIO    = 0.25       # minimum 25 % composite
COMPOSITE_TYP_RATIO    = 0.50       # typical economy target
STUD_DIA_STD_IN        = 0.75       # 3/4" standard headed stud
STUD_DIA_HEAVY_IN      = 0.875      # 7/8" heavy load
SLAB_MIN_ABOVE_DECK_IN = 3.5        # inches above metal deck ribs
DECK_RIB_HEIGHT_IN     = 3.0        # standard 3" deck rib

# ── Diagrid geometric limits (from Appendix C) ───────────────────────────────
DIAGRID_THETA_MIN      = 50.0       # degrees — flattest permitted
DIAGRID_THETA_MAX      = 75.0       # degrees — steepest permitted
DIAGRID_THETA_DEFAULT  = 62.0       # balanced gravity + lateral
DIAGRID_MIN_MEMBERS_AT_NODE = 3     # topology rule

# ── Space-frame / long-span roof ─────────────────────────────────────────────
SPACE_FRAME_DEPTH_SPAN_MIN = 1 / 20
SPACE_FRAME_DEPTH_SPAN_MAX = 1 / 10
SPACE_FRAME_MODULE_MIN_M   = 3.0    # metres
SPACE_FRAME_MODULE_MAX_M   = 6.0    # metres
LONG_SPAN_TRIGGER_FT       = 60.0   # feet — triggers space frame / truss
TRUSS_DEPTH_RATIO_MIN      = 1 / 18
TRUSS_DEPTH_RATIO_MAX      = 1 / 8

# ── RC core (ACI 318-19 §18.10) ──────────────────────────────────────────────
RC_WALL_MIN_THICKNESS_IN   = 12.0   # inches for hw/lw > 2 special wall
RC_LONG_REIN_MIN           = 0.0025 # ρl min each face
RC_LONG_REIN_MAX           = 0.006
RC_BOUNDARY_STRESS_TRIGGER = 0.2    # σ > 0.2·f'c at extreme fibre → boundary element

# ── Shell element selection (Chapter 17) ─────────────────────────────────────
SHELL_THIN_LAMBDA_MIN      = 20.0   # λ = L/t > 20 → Kirchhoff thin
SHELL_THICK_LAMBDA_MAX     = 10.0   # λ < 10 → solid or thick


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MESH GEOMETRY DESCRIPTOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeshDescriptors:
    """All geometry descriptors extracted from a watertight mesh."""
    centroid:           np.ndarray          # (3,) world coords
    bbox_min:           np.ndarray          # (3,)
    bbox_max:           np.ndarray          # (3,)
    height_m:           float
    width_m:            float
    depth_m:            float
    H_W_ratio:          float
    volume_m3:          float
    surface_area_m2:    float
    curvature_type:     str                 # 'synclastic'|'anticlastic'|'flat'|'compound'
    max_mean_curvature: float               # 1/m
    max_gaussian_curv:  float               # 1/m²
    flatness_zones:     list[np.ndarray]    # list of face index arrays
    shaft_candidates:   list[dict]          # [{'centroid': (3,), 'area_m2': float}]
    reentrant_corners:  list[np.ndarray]    # face indices with negative plan curvature
    support_footprint:  np.ndarray          # vertex indices at base (lowest Z cluster)
    volume_discontinuities: bool            # large void or bridging span detected
    void_fraction:      float               # fraction of bounding box volume that is void
    story_count:        int                 # estimated from height / typical floor height
    stories_per_module: int                 # diagrid module suggestion
    aspect_category:    str                 # 'low_wide'|'mid_rise'|'tall'|'super_tall'


def extract_mesh_descriptors(
    vertices: np.ndarray,       # (N, 3) float  — mesh vertex positions, metres
    faces: np.ndarray,          # (M, 3) int    — triangle face vertex indices
    creases: np.ndarray,        # (K, 2) int    — crease edge vertex indices
    floor_height_m: float = 4.0,
    curvature_flat_threshold: float = 0.05,   # 1/m  — faces below this are "flat"
    void_threshold: float = 0.20,             # volume fraction triggering mega-truss
) -> MeshDescriptors:
    """
    Extract all structural geometric descriptors from a watertight mesh.

    Parameters
    ----------
    vertices        : (N, 3) array of vertex positions in metres
    faces           : (M, 3) array of triangle face indices
    creases         : (K, 2) array of crease edge indices (sharp angle changes)
    floor_height_m  : assumed typical floor-to-floor height (default 4.0 m)
    curvature_flat_threshold : mean curvature magnitude below which a face is "flat"
    void_threshold  : fraction of bounding-box volume; above → volume discontinuity

    Returns
    -------
    MeshDescriptors dataclass
    """
    verts = np.asarray(vertices, dtype=float)
    fcs   = np.asarray(faces,    dtype=int)

    # ── Bounding box & basic dimensions ──────────────────────────────────────
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    centroid  = verts.mean(axis=0)
    dims      = bbox_max - bbox_min          # (dx, dy, dz)
    height_m  = float(dims[2])
    width_m   = float(min(dims[0], dims[1]))
    depth_m   = float(max(dims[0], dims[1]))
    H_W       = height_m / max(width_m, 1e-6)

    # ── Bounding-box volume vs mesh volume estimate ───────────────────────────
    bbox_vol  = float(dims[0] * dims[1] * dims[2])
    mesh_vol  = _signed_mesh_volume(verts, fcs)
    void_frac = max(0.0, 1.0 - abs(mesh_vol) / max(bbox_vol, 1e-6))

    # ── Per-face area and face centroids ─────────────────────────────────────
    v0 = verts[fcs[:, 0]]
    v1 = verts[fcs[:, 1]]
    v2 = verts[fcs[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_normals = cross
    face_areas   = np.linalg.norm(cross, axis=1) * 0.5
    surface_area = float(face_areas.sum())
    face_centroids = (v0 + v1 + v2) / 3.0

    # ── Discrete mean curvature (angle-weighted normal deviation) ─────────────
    mean_curv_per_face = _estimate_face_mean_curvature(verts, fcs)
    max_mean_curv = float(np.abs(mean_curv_per_face).max()) if len(mean_curv_per_face) else 0.0

    # ── Gaussian curvature sign → synclastic / anticlastic / compound ─────────
    gauss_curv_per_face = _estimate_face_gaussian_curvature(verts, fcs)
    max_gauss_curv = float(np.abs(gauss_curv_per_face).max()) if len(gauss_curv_per_face) else 0.0
    pos_gauss = (gauss_curv_per_face > 0.01).sum()
    neg_gauss = (gauss_curv_per_face < -0.01).sum()
    if pos_gauss > 0 and neg_gauss > 0:
        curvature_type = "compound"
    elif pos_gauss > neg_gauss:
        curvature_type = "synclastic"
    elif neg_gauss > pos_gauss:
        curvature_type = "anticlastic"
    else:
        curvature_type = "flat"

    # ── Flatness zones ────────────────────────────────────────────────────────
    flat_mask   = np.abs(mean_curv_per_face) < curvature_flat_threshold
    flat_faces  = [np.where(flat_mask)[0]]

    # ── Shaft candidates (low-curvature interior zones near centroid) ─────────
    shaft_candidates = _detect_shaft_candidates(verts, fcs, face_centroids, centroid)

    # ── Re-entrant corners (signed curvature < 0 in plan) ─────────────────────
    reentrant = [np.where(gauss_curv_per_face < -0.05)[0]]

    # ── Support footprint (lowest Z cluster, within 5 % of height) ────────────
    z_base_thresh = bbox_min[2] + 0.05 * height_m
    support_mask  = verts[:, 2] <= z_base_thresh
    support_fp    = np.where(support_mask)[0]

    # ── Story count & aspect category ────────────────────────────────────────
    story_count = max(1, round(height_m / floor_height_m))
    if H_W > 8:
        aspect_cat = "super_tall"
    elif H_W > 4:
        aspect_cat = "tall"
    elif H_W > 1:
        aspect_cat = "mid_rise"
    else:
        aspect_cat = "low_wide"

    # Diagrid module: 2 floors for short, 3–4 for tall
    if story_count <= 10:
        stories_per_module = 2
    elif story_count <= 20:
        stories_per_module = 3
    else:
        stories_per_module = 4

    return MeshDescriptors(
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        height_m=height_m,
        width_m=width_m,
        depth_m=depth_m,
        H_W_ratio=H_W,
        volume_m3=abs(mesh_vol),
        surface_area_m2=surface_area,
        curvature_type=curvature_type,
        max_mean_curvature=max_mean_curv,
        max_gaussian_curv=max_gauss_curv,
        flatness_zones=flat_faces,
        shaft_candidates=shaft_candidates,
        reentrant_corners=reentrant,
        support_footprint=support_fp,
        volume_discontinuities=(void_frac > void_threshold),
        void_fraction=void_frac,
        story_count=story_count,
        stories_per_module=stories_per_module,
        aspect_category=aspect_cat,
    )


def _signed_mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Divergence theorem — signed volume of a closed triangulated mesh."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    signed_vols = np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0
    return float(signed_vols.sum())


def _estimate_face_mean_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Approximate per-face mean curvature as the mean of vertex mean curvatures
    (angle-surplus method — discrete Laplace-Beltrami approximation).
    Returns array of shape (M,), units 1/m.
    """
    n_verts = len(verts)
    vertex_curv = np.zeros(n_verts)
    vertex_area = np.zeros(n_verts)
    for f in faces:
        i, j, k = f
        a = verts[j] - verts[i]; b = verts[k] - verts[i]
        area = np.linalg.norm(np.cross(a, b)) / 2.0
        for v in (i, j, k):
            vertex_curv[v] += area   # placeholder for proper Laplacian
            vertex_area[v] += area / 3.0
    # Very simplified — replace with cotangent formula in production
    safe_area = np.where(vertex_area > 1e-12, vertex_area, 1.0)
    kH = vertex_curv / safe_area - 1.0   # offset so flat → 0
    # Average onto faces
    face_kH = (kH[faces[:, 0]] + kH[faces[:, 1]] + kH[faces[:, 2]]) / 3.0
    # Rescale to physically plausible range 1/m
    max_abs = np.abs(face_kH).max() + 1e-12
    bbox_diag = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    face_kH = face_kH / max_abs * (1.0 / max(bbox_diag, 1.0))
    return face_kH


def _estimate_face_gaussian_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Approximate per-face Gaussian curvature via angle defect at vertices,
    projected back to faces. Units 1/m².
    """
    n_verts = len(verts)
    angle_defect = np.full(n_verts, 2.0 * math.pi)
    vertex_area  = np.zeros(n_verts)
    for f in faces:
        i, j, k = f
        pts = verts[[i, j, k]]
        for local, (a, b, c) in enumerate([(i, j, k), (j, k, i), (k, i, j)]):
            ea = pts[(local+1) % 3] - pts[local]
            eb = pts[(local+2) % 3] - pts[local]
            denom = np.linalg.norm(ea) * np.linalg.norm(eb)
            if denom > 1e-12:
                cos_a = np.clip(np.dot(ea, eb) / denom, -1, 1)
                angle_defect[a] -= math.acos(cos_a)
        area = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])) / 2.0
        for v in (i, j, k):
            vertex_area[v] += area / 3.0
    safe_area = np.where(vertex_area > 1e-12, vertex_area, 1.0)
    kG_vertex = angle_defect / safe_area
    face_kG = (kG_vertex[faces[:, 0]] + kG_vertex[faces[:, 1]] + kG_vertex[faces[:, 2]]) / 3.0
    return face_kG


def _detect_shaft_candidates(
    verts: np.ndarray,
    faces: np.ndarray,
    face_centroids: np.ndarray,
    building_centroid: np.ndarray,
) -> list[dict]:
    """
    Identify interior shaft zones — low-curvature faces whose centroids are
    within 25 % of the plan half-width from the building centroid in XY.
    Returns list of {'centroid': (3,), 'area_m2': float}.
    """
    plan_dist = np.linalg.norm(face_centroids[:, :2] - building_centroid[:2], axis=1)
    plan_radius = np.linalg.norm(verts[:, :2].max(axis=0) - verts[:, :2].min(axis=0)) * 0.25
    inner_mask = plan_dist < plan_radius
    if not inner_mask.any():
        return []
    inner_centroids = face_centroids[inner_mask]
    inner_centroid  = inner_centroids.mean(axis=0)
    v0 = verts[faces[inner_mask, 0]]
    v1 = verts[faces[inner_mask, 1]]
    v2 = verts[faces[inner_mask, 2]]
    inner_area = float((np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5).sum())
    return [{"centroid": inner_centroid, "area_m2": inner_area}]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SYSTEM SELECTION DECISION TREE  (Chapter 19)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemSelection:
    lateral_system:  str
    gravity_system:  str
    has_mega_truss:  bool
    has_outriggers:  bool
    n_outrigger_levels: int
    material:        str    # 'steel_concrete' | 'tcc_hybrid'
    seismic_system:  dict   # from SEISMIC_SYSTEMS
    notes:           list[str] = field(default_factory=list)


def select_structural_system(
    desc: MeshDescriptors,
    sdc: Literal["A", "B", "C", "D", "E", "F"] = "D",
    program: str = "office",    # 'office' | 'residential' | 'assembly' | 'stadium'
) -> SystemSelection:
    """
    MAP mesh descriptors → structural system typology.
    Implements the decision tree of Chapter 19.

    Rules
    -----
    H/W > 8        → super-tall  : diagrid + core + outrigger
    4 ≤ H/W ≤ 8   → tall        : diagrid OR core + SMF/BRBF
    1 ≤ H/W ≤ 4   → mid-rise    : core + composite frame OR space frame
    H/W < 1        → low/wide    : space frame / gridshell
    void_frac > 0.20 → add mega-truss at void level
    compound curvature → hybrid steel shell + concrete wall (MAD-type)
    stories 5–18 + residential/office → evaluate TCC/CLT hybrid
    stories > 18   → steel + concrete composite preferred
    SDC D–F        → special systems required
    """
    notes = []
    hw = desc.H_W_ratio
    stories = desc.story_count
    void = desc.volume_discontinuities
    curv = desc.curvature_type

    # ── Material ──────────────────────────────────────────────────────────────
    if 5 <= stories <= 18 and program in ("residential", "office"):
        material = "tcc_hybrid"
        notes.append("TCC/CLT hybrid evaluated (5–18 stories, residential/office program).")
    else:
        material = "steel_concrete"

    # ── Lateral & gravity system ───────────────────────────────────────────────
    has_outriggers = False
    n_out = 0

    if hw > 8:
        lateral  = "DIAGRID_EBF"
        gravity  = "composite_frame"
        has_outriggers = True
        n_out = max(2, int(hw // 4))
        notes.append(f"Super-tall: diagrid + RC core + {n_out} outrigger levels.")

    elif hw >= 4:
        if curv in ("synclastic", "compound"):
            lateral = "DIAGRID_SCBF"
            gravity = "composite_frame"
            notes.append("Tall + curved façade → diagrid tube preferred.")
        else:
            lateral = "BRBF" if sdc in ("D", "E", "F") else "SCBF"
            gravity = "composite_frame"
            notes.append("Tall + regular → BRBF + core + composite frame.")
        if hw > 6:
            has_outriggers = True
            n_out = 1
            notes.append("Outrigger at mid-height added for H/W > 6.")

    elif hw >= 1:
        if curv in ("compound",):
            lateral = "RC_SPECIAL_WALL"
            gravity = "composite_frame"
            notes.append("Mid-rise compound curvature → hybrid steel shell + RC wall (MAD-type).")
        elif desc.surface_area_m2 / (desc.width_m * desc.depth_m + 1e-6) > 2.5:
            lateral = "RC_SPECIAL_WALL"
            gravity = "space_frame"
            notes.append("Wide low profile + large roof → space frame on RC core.")
        else:
            lateral = "SMF" if sdc in ("D", "E", "F") else "SMF"
            gravity = "composite_frame"
            notes.append("Mid-rise → SMF + composite frame.")

    else:  # H/W < 1  — low/wide
        lateral  = "RC_SPECIAL_WALL"
        gravity  = "space_frame"
        notes.append("Low/wide massing → space frame / gridshell + perimeter RC wall.")

    # ── Mega-truss for voids ──────────────────────────────────────────────────
    has_mega_truss = False
    if void:
        has_mega_truss = True
        notes.append(f"Volume discontinuity detected (void fraction {desc.void_fraction:.1%}) → mega-truss added.")

    # ── SDC override ──────────────────────────────────────────────────────────
    if sdc in ("D", "E", "F"):
        notes.append("SDC D–F: special seismic detailing per AISC 341-22 required.")

    return SystemSelection(
        lateral_system=lateral,
        gravity_system=gravity,
        has_mega_truss=has_mega_truss,
        has_outriggers=has_outriggers,
        n_outrigger_levels=n_out,
        material=material,
        seismic_system=SEISMIC_SYSTEMS.get(lateral, SEISMIC_SYSTEMS["SMF"]),
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD FUNCTIONS  (Chapter 2)
# ══════════════════════════════════════════════════════════════════════════════

def gravity_loads_psf(
    program: str = "office",
    sdl_psf: float = 20.0,
    cladding_psf: float = 15.0,
) -> dict[str, float]:
    """
    Return characteristic gravity load intensities in psf.

    Dead loads are computed by the analysis engine from section geometry +
    material density.  This function returns superimposed and live values.

    ASCE 7-22 §3.1, Table 4.3-1
    """
    live_map = {
        "office":    50.0,
        "residential": 40.0,
        "assembly":  100.0,
        "stadium":   100.0,
        "roof_flat":  20.0,
    }
    L = live_map.get(program, 50.0)
    return {
        "SDL_psf":      sdl_psf,
        "SDL_cladding": cladding_psf,
        "L_psf":        L,
        "Lr_psf":       20.0,   # minimum roof live
    }


def lrfd_factored_load(
    D: float,               # dead load effect (force / moment / stress)
    L: float = 0.0,
    Lr: float = 0.0,
    S: float = 0.0,
    W: float = 0.0,
    E: float = 0.0,
    combo: str = "LC2",
) -> float:
    """
    Compute LRFD factored demand for a single load combination.
    All inputs in consistent units (kips, kip-ft, ksi, etc.).

    Returns
    -------
    Wu : factored demand (same units as inputs)
    """
    if combo == "LC1":
        return 1.4 * D
    elif combo == "LC2":
        return 1.2 * D + 1.6 * L + 0.5 * max(Lr, S)
    elif combo == "LC3":
        return 1.2 * D + 1.6 * max(Lr, S) + max(L, 0.5 * W)
    elif combo == "LC4":
        return 1.2 * D + 1.0 * W + L + 0.5 * max(Lr, S)
    elif combo == "LC5":
        return 0.9 * D + 1.0 * W
    elif combo == "LC6":
        return 1.2 * D + 1.0 * E + L + 0.2 * S
    elif combo == "LC7":
        return 0.9 * D + 1.0 * E
    else:
        raise ValueError(f"Unknown combo '{combo}'. Use LC1–LC7.")


def controlling_lrfd_combo(
    D: float,
    L: float = 0.0,
    Lr: float = 0.0,
    S: float = 0.0,
    W: float = 0.0,
    E: float = 0.0,
) -> tuple[str, float]:
    """
    Iterate all seven LRFD combinations and return (controlling_combo, max_demand).
    """
    combos = ["LC1", "LC2", "LC3", "LC4", "LC5", "LC6", "LC7"]
    demands = [
        lrfd_factored_load(D, L, Lr, S, W, E, c) for c in combos
    ]
    idx = int(np.argmax(demands))
    return combos[idx], demands[idx]


def seismic_special_load_effect(
    QE: float,
    SDS: float,
    D: float,
    system: str = "SMF",
    additive: bool = True,
) -> float:
    """
    Special seismic load effect Em with overstrength (ASCE 7-22 §12.4.3).

    Em = Ω₀ · QE ± 0.2 · SDS · D

    Parameters
    ----------
    QE       : effect due to horizontal seismic forces
    SDS      : design spectral acceleration parameter (short period)
    D        : dead load effect
    system   : key into SEISMIC_SYSTEMS dict
    additive : if True, use + sign (max compression); else - (uplift)
    """
    omega = SEISMIC_SYSTEMS[system]["omega"]
    sign = +1 if additive else -1
    return omega * QE + sign * 0.2 * SDS * D


def base_shear_equivalent_lateral(
    W_total_kips: float,
    Cs: float,
) -> float:
    """
    V = Cs · W  (ASCE 7-22 §12.8.1)

    Cs is the seismic response coefficient (obtained from site parameters,
    R-factor, and Ie).  This function simply applies the formula.
    """
    return Cs * W_total_kips


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MEMBER SIZING FUNCTIONS  (Chapter 11)
# ══════════════════════════════════════════════════════════════════════════════

def required_beam_section_modulus_in3(
    Mu_kip_ft: float,
    phi_b: float = 0.90,
    Fy_ksi: float = Fy_A992,
) -> float:
    """
    Minimum plastic section modulus Zx for a compact W-shape beam (full lateral bracing).
    φb·Mn = φb·Fy·Zx ≥ Mu  →  Zx_req = Mu / (φb · Fy)

    Parameters
    ----------
    Mu_kip_ft : factored bending demand, kip-ft
    phi_b     : 0.90 for LRFD bending
    Fy_ksi    : yield stress, ksi

    Returns
    -------
    Zx_req    : required plastic section modulus, in³
    """
    Mu_kip_in = Mu_kip_ft * 12.0
    return Mu_kip_in / (phi_b * Fy_ksi)


def required_column_area_in2(
    Pu_kips: float,
    phi_c: float = 0.90,
    Fy_ksi: float = Fy_A992,
    Fcr_factor: float = 0.877,   # Euler critical stress factor for KL/r ≈ 60 typical col
) -> float:
    """
    Minimum gross area for a column in pure compression.
    φc·Pn ≥ Pu  →  A_req = Pu / (φc · Fcr)

    Fcr_factor accounts for stability (≈ 0.877 for intermediate slenderness).
    For exact Fcr, use AISC 360 §E3 with known KL/r.

    Returns
    -------
    A_req : required gross cross-section area, in²
    """
    Fcr = Fcr_factor * Fy_ksi
    return Pu_kips / (phi_c * Fcr)


def aisc_360_interaction_H1(
    Pr: float, Pc: float,
    Mrx: float, Mcx: float,
    Mry: float = 0.0, Mcy: float = 1e9,
) -> float:
    """
    AISC 360-22 §H1 bilinear interaction ratio (DCR) for combined
    axial + biaxial bending.

    If Pr/Pc ≥ 0.2:   DCR = Pr/Pc + (8/9)(Mrx/Mcx + Mry/Mcy)
    If Pr/Pc < 0.2:   DCR = Pr/(2·Pc) + (Mrx/Mcx + Mry/Mcy)

    Parameters
    ----------
    Pr  : required axial force (factored), kips  (positive = compression)
    Pc  : available axial capacity = φc·Pn, kips
    Mrx : required major-axis moment, kip-ft
    Mcx : available major-axis moment capacity = φb·Mpx (or Mn), kip-ft
    Mry : required minor-axis moment, kip-ft
    Mcy : available minor-axis capacity, kip-ft

    Returns
    -------
    DCR : demand/capacity ratio  (≤ 1.0 = pass, > 1.0 = fail)
    """
    ratio = abs(Pr) / max(abs(Pc), 1e-9)
    bending = abs(Mrx) / max(abs(Mcx), 1e-9) + abs(Mry) / max(abs(Mcy), 1e-9)
    if ratio >= 0.2:
        return ratio + (8.0 / 9.0) * bending
    else:
        return ratio / 2.0 + bending


def column_fcr_aisc360_E3(
    KL_over_r: float,
    Fy_ksi: float = Fy_A992,
    E_ksi: float = E_STEEL,
) -> float:
    """
    Critical stress Fcr for compression members per AISC 360-22 §E3.

    If KL/r ≤ 4.71√(E/Fy):   Fcr = [0.658^(Fy/Fe)] · Fy   (inelastic)
    Else:                      Fcr = 0.877 · Fe              (elastic)

    Parameters
    ----------
    KL_over_r : effective slenderness ratio (dimensionless)
    Fy_ksi    : yield stress, ksi
    E_ksi     : Young's modulus, ksi

    Returns
    -------
    Fcr : critical compressive stress, ksi
    """
    limit = 4.71 * math.sqrt(E_ksi / Fy_ksi)
    Fe    = math.pi**2 * E_ksi / max(KL_over_r, 1e-3)**2
    if KL_over_r <= limit:
        Fcr = (0.658 ** (Fy_ksi / Fe)) * Fy_ksi
    else:
        Fcr = 0.877 * Fe
    return min(Fcr, Fy_ksi)   # cap at yield


def available_axial_capacity_kips(
    A_in2: float,
    KL_ft: float,
    r_in: float,
    Fy_ksi: float = Fy_A992,
    phi_c: float = 0.90,
) -> float:
    """
    φc·Pn for a compression member per AISC 360-22 §E3.

    Parameters
    ----------
    A_in2  : gross cross-section area, in²
    KL_ft  : effective unbraced length, feet
    r_in   : governing radius of gyration, inches
    """
    KL_r  = KL_ft * 12.0 / max(r_in, 0.01)
    Fcr   = column_fcr_aisc360_E3(KL_r, Fy_ksi)
    Pn    = Fcr * A_in2
    return phi_c * Pn


def available_moment_capacity_kip_ft(
    Zx_in3: float,
    Lb_ft: float,
    Lp_ft: float,
    Lr_ft: float,
    Mp_kip_ft: float,
    Fy_ksi: float = Fy_A992,
    phi_b: float = 0.90,
) -> float:
    """
    φb·Mn for lateral-torsional buckling (LTB) of W-shape beam per AISC 360-22 §F2.

    Regions:
      Lb ≤ Lp  → plastic:   Mn = Mp
      Lp < Lb ≤ Lr → inelastic:  Mn = Mp - (Mp - 0.7·Fy·Sx)·(Lb-Lp)/(Lr-Lp)
      Lb > Lr  → elastic LTB (not implemented here; use full §F2)

    Parameters
    ----------
    Zx_in3    : plastic section modulus, in³
    Lb_ft     : unbraced length, ft
    Lp_ft     : limiting unbraced length for plastic moment, ft  (from tables)
    Lr_ft     : limiting unbraced length for inelastic LTB, ft
    Mp_kip_ft : plastic moment capacity = Fy·Zx / 12, kip-ft
    """
    Sx_in3 = Zx_in3 / 1.15   # approximate: Sx ≈ Zx / shape_factor
    if Lb_ft <= Lp_ft:
        Mn = Mp_kip_ft
    elif Lb_ft <= Lr_ft:
        Mn = Mp_kip_ft - (Mp_kip_ft - 0.7 * Fy_ksi * Sx_in3 / 12.0) * (
            (Lb_ft - Lp_ft) / max(Lr_ft - Lp_ft, 1e-6)
        )
    else:
        # Elastic LTB — conservative lower bound
        Mn = 0.7 * Fy_ksi * Sx_in3 / 12.0
    return phi_b * max(Mn, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DIAGRID GEOMETRY GENERATION  (Chapter 4 & 20)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagridNode:
    id:     str
    xyz:    np.ndarray   # (3,) metres
    floor_level: bool    # True if node sits at a floor Z-level

@dataclass
class DiagridMember:
    id:       str
    node_i:   str
    node_j:   str
    theta_deg: float     # inclination from horizontal
    length_m:  float
    role:     str        # 'diagonal' | 'ring_beam' | 'core_link'

@dataclass
class DiagridLayout:
    nodes:       list[DiagridNode]
    members:     list[DiagridMember]
    theta_deg:   float
    module_h_m:  float
    bay_count:   int
    section_family: str   # e.g. 'HSS10-16' | 'buildup_box'
    node_type:   str      # 'plate_welded' | 'cast_steel'


def diagrid_theta_from_HW(H_W_ratio: float) -> float:
    """
    Select optimal diagrid inclination angle θ (degrees from horizontal)
    based on building slenderness, per Appendix C parametric matrix.

    Domain: H/W < 3  → 65–75°
            3–6      → 60–70°
            6–8      → 55–65°
            > 8      → 50–60°
    Returns midpoint of range.
    """
    if H_W_ratio < 3:
        return 70.0
    elif H_W_ratio <= 6:
        return 65.0
    elif H_W_ratio <= 8:
        return 60.0
    else:
        return 55.0


def diagrid_module_height_m(
    floor_height_m: float,
    stories_per_module: int,
) -> float:
    """
    H_m = floor_height · N_floors_per_module
    Typical N = 2 (short), 3 (mid), 4 (tall).
    """
    return floor_height_m * stories_per_module


def diagrid_bay_count(
    perimeter_m: float,
    module_height_m: float,
    theta_deg: float,
) -> int:
    """
    Number of diagrid bays along the perimeter.

    Each bay has a horizontal width  w = H_m / tan(θ),
    so bay count = perimeter / w.
    Rounded to nearest even integer (symmetry).

    Parameters
    ----------
    perimeter_m    : building perimeter at mid-height, metres
    module_height_m: vertical height of one diagrid module, metres
    theta_deg      : inclination angle, degrees

    Returns
    -------
    n_bays : integer (minimum 4)
    """
    theta_rad = math.radians(theta_deg)
    w = module_height_m / math.tan(theta_rad)   # horizontal projection of one diagonal
    n = max(4, round(perimeter_m / w))
    return n if n % 2 == 0 else n + 1


def diagrid_diagonal_length_m(
    module_height_m: float,
    theta_deg: float,
) -> float:
    """
    True 3D length of one diagrid diagonal member.
    L_diag = H_m / sin(θ)
    """
    return module_height_m / math.sin(math.radians(theta_deg))


def diagrid_axial_force_gravity(
    floor_tributary_load_kN_m2: float,
    floor_bay_area_m2: float,
    theta_deg: float,
    n_diagonals_per_node: int = 2,
) -> float:
    """
    Approximate axial force in a diagrid diagonal due to gravity.

    Each diagonal carries a tributary gravity load P_grav from its floor bay.
    The vertical component of the diagonal force = P_grav / n_diagonals.
    Axial force = vertical_component / sin(θ).

    Parameters
    ----------
    floor_tributary_load_kN_m2 : factored (wu) floor load, kN/m²
    floor_bay_area_m2          : tributary plan area per diagonal, m²
    theta_deg                  : inclination, degrees
    n_diagonals_per_node       : diagonals sharing the node (typically 2–4)

    Returns
    -------
    N_diag_kN : axial force (positive = compression), kN
    """
    P_grav = floor_tributary_load_kN_m2 * floor_bay_area_m2
    N_diag = P_grav / (n_diagonals_per_node * math.sin(math.radians(theta_deg)))
    return N_diag


def diagrid_axial_force_lateral(
    V_lateral_kN: float,
    theta_deg: float,
    n_perimeter_diagonals: int,
) -> float:
    """
    Approximate axial force in a diagrid diagonal due to lateral (wind/seismic) shear V.

    Each diagonal's horizontal component = N·cos(θ).
    Sum of all diagonals' horizontal components = V_lateral
    → N_lateral = V_lateral / (n_diagonals · cos(θ))

    Parameters
    ----------
    V_lateral_kN         : total story shear, kN
    theta_deg            : inclination, degrees
    n_perimeter_diagonals: number of active diagonals at the story

    Returns
    -------
    N_lateral_kN : lateral axial force component, kN
    """
    return V_lateral_kN / (n_perimeter_diagonals * math.cos(math.radians(theta_deg)))


def diagrid_required_HSS_diameter_mm(
    N_total_kN: float,
    KL_m: float,
    Fy_MPa: float = 317.0,   # A500 Gr.C in MPa
    phi_c: float = 0.90,
    safety_factor: float = 1.1,
) -> float:
    """
    Estimate minimum CHS (round HSS) outer diameter for a diagrid diagonal
    under combined compression.

    Assumes D/t ≈ 30 (typical for CHS), r ≈ D·√2/4 for thin-walled tube.

    Returns
    -------
    D_min_mm : minimum outside diameter, mm
    """
    N_kips = N_total_kN * safety_factor * 0.2248   # kN → kips
    KL_ft  = KL_m * 3.2808

    # Iterative: guess D, compute r = D/√8, check Pn ≥ N_kips
    for D_in in np.arange(4, 24, 0.5):
        t_in = D_in / 30.0
        A_in2 = math.pi * (D_in**2 - (D_in - 2*t_in)**2) / 4.0
        r_in  = math.sqrt(D_in**2 + (D_in - 2*t_in)**2) / 4.0
        KL_r  = KL_ft * 12.0 / max(r_in, 0.01)
        if KL_r > SLENDERNESS_MAX:
            continue
        Fcr  = column_fcr_aisc360_E3(KL_r, Fy_ksi=Fy_MPa * 0.14504)  # MPa→ksi
        Pn   = phi_c * Fcr * A_in2
        if Pn >= N_kips:
            return D_in * 25.4   # inches → mm
    return 24.0 * 25.4  # fallback max


def generate_diagrid_layout(
    desc: MeshDescriptors,
    floor_height_m: float = 4.0,
) -> DiagridLayout:
    """
    Procedural diagrid skeleton generation from mesh descriptors.
    Implements Algorithm 20.1.

    Returns a DiagridLayout with node positions and member connectivity.
    NOTE: This is a parametric skeleton generator. Actual mesh-projected positions
    require the caller to project nodes onto the mesh surface.
    """
    theta  = diagrid_theta_from_HW(desc.H_W_ratio)
    H_m    = diagrid_module_height_m(floor_height_m, desc.stories_per_module)
    perim  = 2 * (desc.width_m + desc.depth_m)
    n_bays = diagrid_bay_count(perim, H_m, theta)
    L_diag = diagrid_diagonal_length_m(H_m, theta)

    # Section family from Appendix C
    if desc.height_m < 30:
        sec_fam = "HSS8-12"
        node_tp = "plate_welded"
    elif desc.height_m < 90:
        sec_fam = "HSS10-16"
        node_tp = "plate_welded_or_cast"
    elif desc.height_m < 180:
        sec_fam = "buildup_box_W14"
        node_tp = "cast_steel"
    else:
        sec_fam = "buildup_box_mega"
        node_tp = "cast_steel_AESS4"

    nodes   : list[DiagridNode]   = []
    members : list[DiagridMember] = []

    n_floors = desc.story_count
    bay_width = perim / n_bays

    node_id = 0
    member_id = 0

    # Generate nodes on a rolled-out perimeter grid, then project back
    for iz in range(int(n_floors / desc.stories_per_module) + 1):
        z = iz * H_m + desc.bbox_min[2]
        is_floor = True
        for ibay in range(n_bays):
            # Stagger diagonals: even z-level offset by half bay
            x_offset = bay_width * ibay + (bay_width / 2.0 if iz % 2 == 0 else 0.0)
            # Map to perimeter XY (rectangular approximation)
            cx, cy = _perimeter_xy(x_offset, perim, desc)
            nid = f"DG_{node_id:04d}"
            nodes.append(DiagridNode(id=nid, xyz=np.array([cx, cy, z]), floor_level=is_floor))
            node_id += 1

    # Connect diagonals between levels (simplified: nearest-neighbour)
    # A full implementation would use the mesh surface projection
    for m_idx in range(len(nodes) // 2):
        ni = nodes[m_idx]
        nj_idx = (m_idx + n_bays // 2) % len(nodes)
        nj = nodes[nj_idx]
        length = float(np.linalg.norm(ni.xyz - nj.xyz))
        members.append(DiagridMember(
            id=f"DG_MEM_{member_id:04d}",
            node_i=ni.id, node_j=nj.id,
            theta_deg=theta,
            length_m=length,
            role="diagonal",
        ))
        member_id += 1

    return DiagridLayout(
        nodes=nodes, members=members,
        theta_deg=theta, module_h_m=H_m, bay_count=n_bays,
        section_family=sec_fam, node_type=node_tp,
    )


def _perimeter_xy(dist_along: float, total_perim: float, desc: MeshDescriptors):
    """Map a 1D distance along perimeter to 2D XY for a rectangular footprint."""
    cx0, cy0 = desc.bbox_min[0], desc.bbox_min[1]
    W, D = desc.width_m, desc.depth_m
    d = dist_along % total_perim
    if d < W:
        return cx0 + d, cy0
    d -= W
    if d < D:
        return cx0 + W, cy0 + d
    d -= D
    if d < W:
        return cx0 + W - d, cy0 + D
    d -= W
    return cx0, cy0 + D - d


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPACE FRAME / LONG-SPAN GENERATION  (Chapter 5 & 20)
# ══════════════════════════════════════════════════════════════════════════════

def space_frame_depth_m(span_m: float) -> float:
    """
    Space frame depth D = span / ratio.
    Ratio in range [10, 20]; choose 15 as default.
    Domain: SPACE_FRAME_DEPTH_SPAN_MIN ≤ D/span ≤ SPACE_FRAME_DEPTH_SPAN_MAX
    """
    ratio = 15.0
    return max(span_m * SPACE_FRAME_DEPTH_SPAN_MIN,
               min(span_m * SPACE_FRAME_DEPTH_SPAN_MAX,
                   span_m / ratio))


def space_frame_module_m(
    cladding_module_m: float = 1.5,
) -> float:
    """
    Space-frame grid module size constrained by cladding system.
    Domain: SPACE_FRAME_MODULE_MIN_M ≤ module ≤ SPACE_FRAME_MODULE_MAX_M
    """
    return max(SPACE_FRAME_MODULE_MIN_M, min(SPACE_FRAME_MODULE_MAX_M, cladding_module_m * 2))


def truss_depth_m(span_m: float, heavy: bool = False) -> float:
    """
    Roof / floor truss depth.
    D/span = 1/18 (light) to 1/8 (heavy transfer).
    """
    ratio = 8.0 if heavy else 15.0
    return span_m / ratio


def ring_beam_required(
    support_type: Literal["discrete_column", "wall", "continuous_perimeter"],
    span_m: float,
) -> bool:
    """
    Rule: a perimeter ring beam is ALWAYS required for space frames and
    gridshells to collect horizontal thrust.
    Returns True if ring beam must be generated.
    """
    return True   # unconditional per Chapter 5 §5.3


def horizontal_thrust_kN(
    V_kN_m2: float,
    span_m: float,
    rise_m: float,
) -> float:
    """
    Horizontal thrust at support for a parabolic arch / shell under uniform load.
    H = w·L² / (8·f)   where f = rise, L = span.

    Parameters
    ----------
    V_kN_m2 : uniform vertical load intensity, kN/m²
    span_m  : horizontal span, m
    rise_m  : structural rise (sag) of the arch, m

    Returns
    -------
    H_kN : horizontal thrust per unit width, kN/m
    """
    return V_kN_m2 * span_m**2 / (8.0 * max(rise_m, 0.01))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RC CORE SIZING  (Chapter 14)
# ══════════════════════════════════════════════════════════════════════════════

def core_wall_thickness_mm(height_m: float, special_wall: bool = True) -> float:
    """
    Minimum core wall thickness t = max(300 mm, H_building / 400).
    ACI 318-19 §18.10.2.3 sets 300 mm (12") for hw/lw > 2 special shear walls.

    Returns thickness in mm.
    """
    t_min_mm = RC_WALL_MIN_THICKNESS_IN * 25.4
    t_height = height_m * 1000.0 / 400.0
    return max(t_min_mm, t_height)


def boundary_element_required(
    sigma_extreme_ksi: float,
    fc_ksi: float = fc_CORE,
) -> bool:
    """
    ACI 318-19 §18.10.6 stress-based trigger for special boundary elements.
    Required if σ > 0.2·f'c at extreme compression fibre under seismic.
    """
    return sigma_extreme_ksi > RC_BOUNDARY_STRESS_TRIGGER * fc_ksi


def coupling_beam_type(
    ln_m: float,
    d_m: float,
) -> str:
    """
    ACI 318-19 §18.10.7 coupling beam reinforcement type.

    ln/d ≤ 2  : diagonal reinforcement required (high shear)
    2 < ln/d ≤ 4 : conventional RC with diagonal check
    ln/d > 4  : conventional RC

    Parameters
    ----------
    ln_m : clear span of coupling beam, m
    d_m  : effective depth, m
    """
    ratio = ln_m / max(d_m, 0.01)
    if ratio <= 2.0:
        return "diagonal_reinf"
    elif ratio <= 4.0:
        return "conventional_with_diagonal_check"
    else:
        return "conventional_RC"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — COMPOSITE BEAM FUNCTIONS  (Chapter 13)
# ══════════════════════════════════════════════════════════════════════════════

def effective_slab_width_m(
    span_m: float,
    beam_spacing_m: float,
    edge_distance_m: float = 1e9,
) -> float:
    """
    Effective slab width beff for composite beam per AISC 360-22 §I3.1a.
    beff = min(span/8 each side, beam_spacing/2, edge_distance)
    Total = sum of each side.
    """
    each_side = min(span_m / 8.0, beam_spacing_m / 2.0, edge_distance_m)
    return 2.0 * each_side


def shear_stud_capacity_kips(
    stud_dia_in: float = STUD_DIA_STD_IN,
    fc_ksi: float = fc_CORE,
    Ec_ksi: float = Ec_CONCRETE / 1000.0,
) -> float:
    """
    Nominal strength of a headed shear stud Qn per AISC 360-22 §I8.2a.

    Qn = 0.5 · Asa · √(f'c · Ec)  ≤  Rg · Rp · Asa · Fu

    For solid slab (no deck): Rg=Rp=1.0.
    Fu for A108 stud = 65 ksi.

    Parameters
    ----------
    stud_dia_in : stud shank diameter, inches
    fc_ksi      : concrete compressive strength, ksi
    Ec_ksi      : concrete elastic modulus, ksi

    Returns
    -------
    Qn : nominal stud capacity, kips
    """
    Asa = math.pi * (stud_dia_in / 2.0)**2   # stud shank area, in²
    Fu_stud = 65.0  # ksi A108
    Qn_formula = 0.5 * Asa * math.sqrt(fc_ksi * Ec_ksi)
    Qn_limit   = 1.0 * 0.75 * Asa * Fu_stud  # Rg=Rp=1.0 for solid slab
    return min(Qn_formula, Qn_limit)


def required_stud_count(
    C_prime_kips: float,
    Qn_kips: float,
    composite_ratio: float = COMPOSITE_TYP_RATIO,
) -> int:
    """
    Number of shear studs needed for partial composite action.
    n = ρ · C' / Qn   (per half-span, both flanges)
    Minimum 25 % composite (AISC 360 Commentary §I3).
    """
    eff_ratio = max(COMPOSITE_MIN_RATIO, min(1.0, composite_ratio))
    n = math.ceil(eff_ratio * C_prime_kips / max(Qn_kips, 0.1))
    return max(1, n)


def slab_thickness_m(
    span_m: float,
    slab_type: str = "composite",
) -> float:
    """
    Initial slab thickness estimate from span / depth ratio (Chapter 14 §14.2).

    composite    : t = span / 36
    PT_flat      : t = span / 30
    two_way_flat : t = span / 28
    waffle       : t = span / 22
    """
    ratios = {
        "composite":    36.0,
        "PT_flat":      30.0,
        "two_way_flat": 28.0,
        "waffle":       22.0,
    }
    ratio = ratios.get(slab_type, 36.0)
    t = span_m / ratio
    # Convert to m, enforce minimums
    min_t = 0.089   # 3.5 in above deck
    return max(t, min_t)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SHELL ELEMENT SELECTION  (Chapter 17)
# ══════════════════════════════════════════════════════════════════════════════

ShellType = Literal["thin_kirchhoff", "thick_mindlin", "3d_solid"]

def select_shell_element(
    L_local_m: float,
    t_plate_m: float,
    near_support: bool = False,
    near_opening: bool = False,
    high_curvature: bool = False,
    load_discontinuity: bool = False,
    mesh_distortion_ratio: float = 1.0,
) -> ShellType:
    """
    Select finite element shell formulation per AISC 360 / Chapter 17 logic.

    λ = L_local / t_plate

    λ > 20        → thin (Kirchhoff)
    10 ≤ λ ≤ 20  AND (near support | opening | stiffener) → thick (Mindlin)
    λ < 10        → thick or 3D solid
    high_curvature AND load_discontinuity → override to thick
    mesh_distortion_ratio > 5:1 → avoid thick; use thin + refine

    Returns element type string.
    """
    lam = L_local_m / max(t_plate_m, 1e-6)

    if mesh_distortion_ratio > 5.0:
        return "thin_kirchhoff"

    if high_curvature and load_discontinuity:
        return "thick_mindlin"

    if lam > SHELL_THIN_LAMBDA_MIN:
        return "thin_kirchhoff"
    elif SHELL_THICK_LAMBDA_MAX <= lam <= SHELL_THIN_LAMBDA_MIN:
        if near_support or near_opening:
            return "thick_mindlin"
        return "thin_kirchhoff"
    else:
        if lam < SHELL_THICK_LAMBDA_MAX:
            return "3d_solid"
        return "thick_mindlin"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CREASE / DISCONTINUITY HANDLING  (Chapter 16 §16.2)
# ══════════════════════════════════════════════════════════════════════════════

def crease_stiffening_section(
    span_across_crease_m: float,
    crease_angle_deg: float,
    default_beam_depth_m: float,
) -> float:
    """
    At a crease or ridge (sharp angle change), the bending concentration
    requires a deeper member or plate rib.

    Rule: required depth = default_depth × crease_amplification_factor
    crease_amplification_factor = 1 + (180 - crease_angle) / 180
      (crease_angle = dihedral angle at crease in degrees; 180° = flat)

    Returns required section depth in metres.
    """
    amp = 1.0 + (180.0 - crease_angle_deg) / 180.0
    return default_beam_depth_m * amp


def high_valence_node_zone(
    n_members: int,
    typical_member_length_m: float,
) -> float:
    """
    At a high-valence vertex (many members meeting), model as a small
    rigid zone of radius r_rigid.

    r_rigid ≈ 0.05 × L_member (5 % of member length).
    Returns radius of rigid zone in metres.
    """
    return 0.05 * typical_member_length_m


def free_edge_beam_depth_m(
    edge_span_m: float,
    edge_load_kN_m: float,
    Fy_ksi: float = Fy_A992,
    phi_b: float = 0.90,
) -> float:
    """
    Minimum depth of an edge beam at a free mesh boundary to prevent
    excessive deformation.

    Treat edge beam as a simply supported beam:
    Mu = w·L²/8  → Zx_req = Mu / (φb·Fy)
    Estimate depth from Zx using d ≈ (Zx / 0.3)^(1/3) (empirical fit for W-shapes).

    Parameters
    ----------
    edge_span_m    : span between support nodes along the edge, m
    edge_load_kN_m : distributed load on edge beam, kN/m

    Returns
    -------
    d_min_m : minimum beam depth, metres
    """
    Mu_kN_m  = edge_load_kN_m * edge_span_m**2 / 8.0
    Mu_kip_in = Mu_kN_m * 0.7376 * 12.0   # kN·m → kip·in
    Zx_req_in3 = Mu_kip_in / (phi_b * Fy_ksi)
    d_in = (Zx_req_in3 / 0.30) ** (1.0 / 3.0)
    return d_in * 0.0254   # in → m


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CODE CHECK CONSTRAINT FUNCTIONS  (Chapter 21)
# ══════════════════════════════════════════════════════════════════════════════

StatusType = Literal["pass", "warn", "fail"]

@dataclass
class CodeCheckResult:
    element_id:       str
    check_type:       str
    controlling_combo: str
    DCR:              float
    status:           StatusType
    message:          str


def check_member_interaction(
    element_id: str,
    Pr: float, Pc: float,
    Mrx: float, Mcx: float,
    Mry: float = 0.0, Mcy: float = 1e9,
    combo: str = "LC6",
) -> CodeCheckResult:
    """AISC 360-22 §H1 combined axial + bending check."""
    DCR = aisc_360_interaction_H1(Pr, Pc, Mrx, Mcx, Mry, Mcy)
    if DCR > 1.0:
        status, msg = "fail", f"H1 interaction DCR={DCR:.3f} > 1.0 — OVERLOADED"
    elif DCR > 0.9:
        status, msg = "warn", f"H1 interaction DCR={DCR:.3f} > 0.90 — near limit"
    else:
        status, msg = "pass", f"H1 interaction DCR={DCR:.3f} — OK"
    return CodeCheckResult(element_id, "interaction_H1", combo, DCR, status, msg)


def check_slenderness(
    element_id: str,
    KL_r: float,
    combo: str = "LC2",
) -> CodeCheckResult:
    """AISC 360 §E2 slenderness limit KL/r ≤ 200."""
    DCR = KL_r / SLENDERNESS_MAX
    if KL_r > SLENDERNESS_MAX:
        status, msg = "fail", f"KL/r={KL_r:.1f} > 200 — FAILS slenderness limit"
    elif KL_r > 180:
        status, msg = "warn", f"KL/r={KL_r:.1f} > 180 — approaching limit"
    else:
        status, msg = "pass", f"KL/r={KL_r:.1f} ≤ 200 — OK"
    return CodeCheckResult(element_id, "slenderness_E2", combo, DCR, status, msg)


def check_beam_deflection(
    element_id: str,
    delta_LL_m: float,
    span_m: float,
    long_span_roof: bool = False,
) -> CodeCheckResult:
    """IBC 2024 Table 1604.3 beam deflection check."""
    limit = DEFLECTION_ROOF_LL if long_span_roof else DEFLECTION_LL_LIMIT
    delta_limit = span_m * limit
    DCR = delta_LL_m / max(delta_limit, 1e-9)
    check = "deflection_LL_roof" if long_span_roof else "deflection_LL"
    ratio_str = "L/480" if long_span_roof else "L/360"
    if delta_LL_m > delta_limit:
        status = "fail"
        msg = f"δ_LL={delta_LL_m*1000:.1f}mm > {ratio_str}={delta_limit*1000:.1f}mm — FAILS"
    elif DCR > 0.9:
        status = "warn"
        msg = f"δ_LL={delta_LL_m*1000:.1f}mm — near {ratio_str} limit"
    else:
        status = "pass"
        msg = f"δ_LL={delta_LL_m*1000:.1f}mm ≤ {ratio_str} — OK"
    return CodeCheckResult(element_id, check, "gravity", DCR, status, msg)


def check_story_drift(
    element_id: str,
    delta_m: float,
    h_sx_m: float,
    sdc: str = "D",
    use_type: str = "office",
) -> CodeCheckResult:
    """ASCE 7-22 Table 12.12-1 seismic story drift check."""
    limit = DRIFT_SEISMIC_SDC_DF
    ratio = delta_m / max(h_sx_m, 0.01)
    DCR   = ratio / limit
    if ratio > limit:
        status = "fail"
        msg = f"Drift Δ/h={ratio:.4f} > {limit:.3f} — FAILS SDC {sdc}"
    elif DCR > 0.9:
        status = "warn"
        msg = f"Drift Δ/h={ratio:.4f} — near 0.020 limit"
    else:
        status = "pass"
        msg = f"Drift Δ/h={ratio:.4f} ≤ {limit:.3f} — OK"
    return CodeCheckResult(element_id, "story_drift", "LC6", DCR, status, msg)


def check_composite_stud_capacity(
    element_id: str,
    Qn_n_kips: float,    # Qn × n (total stud capacity)
    C_prime_kips: float,  # required composite shear force
    combo: str = "LC2",
) -> CodeCheckResult:
    """AISC 360 §I3 composite shear stud capacity check."""
    DCR = C_prime_kips / max(Qn_n_kips, 1e-3)
    if Qn_n_kips < C_prime_kips:
        status = "fail"
        msg = f"Stud capacity {Qn_n_kips:.1f} kips < required {C_prime_kips:.1f} kips — FAILS"
    else:
        status = "pass"
        msg = f"Stud capacity OK (ratio={DCR:.3f})"
    return CodeCheckResult(element_id, "stud_capacity", combo, DCR, status, msg)


def check_rc_wall_DCR(
    element_id: str,
    Mu_kip_ft: float, phi_Mn_kip_ft: float,
    Vu_kips: float,   phi_Vn_kips: float,
    combo: str = "LC6",
) -> CodeCheckResult:
    """ACI 318-19 §18.10 combined flexure + shear check for RC shear wall."""
    DCR_M = Mu_kip_ft  / max(phi_Mn_kip_ft, 1e-3)
    DCR_V = Vu_kips    / max(phi_Vn_kips,   1e-3)
    DCR   = max(DCR_M, DCR_V)
    if DCR > 1.0:
        status = "fail"
        msg = f"RC wall DCR={DCR:.3f} (M={DCR_M:.3f}, V={DCR_V:.3f}) — FAILS"
    elif DCR > 0.9:
        status = "warn"
        msg = f"RC wall DCR={DCR:.3f} — near limit"
    else:
        status = "pass"
        msg = f"RC wall DCR={DCR:.3f} — OK"
    return CodeCheckResult(element_id, "rc_wall_ACI318", combo, DCR, status, msg)


def check_punching_shear(
    element_id: str,
    vu_ksi: float,
    fc_ksi: float = fc_CORE,
    phi: float = 0.75,
    lambda_: float = 1.0,   # 1.0 for normal-weight concrete
    combo: str = "LC2",
) -> CodeCheckResult:
    """
    ACI 318-19 punching shear at flat slab columns.
    φ·vc = φ · 4 · λ · √f'c  (in psi → convert to ksi)

    vu ≤ φ·vc required.
    """
    phi_vc_ksi = phi * 4.0 * lambda_ * math.sqrt(fc_ksi * 1000) / 1000.0
    DCR = vu_ksi / max(phi_vc_ksi, 1e-6)
    if vu_ksi > phi_vc_ksi:
        status = "fail"
        msg = f"Punching shear vu={vu_ksi:.4f} ksi > φvc={phi_vc_ksi:.4f} ksi — ADD shear reinforcement"
    elif DCR > 0.9:
        status = "warn"
        msg = f"Punching shear near limit DCR={DCR:.3f}"
    else:
        status = "pass"
        msg = f"Punching shear DCR={DCR:.3f} — OK"
    return CodeCheckResult(element_id, "punching_shear_ACI", combo, DCR, status, msg)


def check_diagrid_node(
    element_id: str,
    member_forces_kN: list[tuple[float, float, float]],   # [(Fx, Fy, Fz), ...]
    weld_capacity_kN: float,
    bolt_capacity_kN: float,
    combo: str = "LC6",
) -> CodeCheckResult:
    """
    Diagrid node equilibrium check: ΣF = 0 at node in all DOFs,
    and weld / bolt capacity ≥ max resolved force.

    Parameters
    ----------
    member_forces_kN  : list of (Fx, Fy, Fz) force vectors arriving at node
    weld_capacity_kN  : total weld group capacity at node
    bolt_capacity_kN  : total bolt group capacity at node
    """
    F_total = np.sum(np.array(member_forces_kN), axis=0)
    imbalance = float(np.linalg.norm(F_total))
    max_force = max((np.linalg.norm(f) for f in member_forces_kN), default=0.0)
    capacity  = min(weld_capacity_kN, bolt_capacity_kN)
    DCR = max_force / max(capacity, 1e-3)
    if imbalance > 1e-3 * max_force:
        return CodeCheckResult(element_id, "diagrid_node_equilibrium", combo, DCR,
                               "fail", f"Node imbalance={imbalance:.2f} kN — CHECK connectivity")
    if DCR > 1.0:
        return CodeCheckResult(element_id, "diagrid_node_capacity", combo, DCR,
                               "fail", f"Node DCR={DCR:.3f} > 1.0 — OVERLOADED")
    elif DCR > 0.9:
        return CodeCheckResult(element_id, "diagrid_node_capacity", combo, DCR,
                               "warn", f"Node DCR={DCR:.3f} near limit")
    return CodeCheckResult(element_id, "diagrid_node_capacity", combo, DCR,
                           "pass", f"Node DCR={DCR:.3f} — OK")


def check_tcc_deflection(
    element_id: str,
    delta_elastic_m: float,
    span_m: float,
    kdef: float = 0.8,   # long-term creep factor (NDS guidance)
) -> CodeCheckResult:
    """
    TCC beam long-term deflection: δ_total = δ_elastic × (1 + kdef) ≤ L/240.
    Chapter 15 §15.3.
    """
    delta_total = delta_elastic_m * (1.0 + kdef)
    limit = span_m * DEFLECTION_TL_LIMIT
    DCR = delta_total / max(limit, 1e-9)
    if delta_total > limit:
        status = "fail"
        msg = f"TCC δ_total={delta_total*1000:.1f}mm > L/240={limit*1000:.1f}mm — INCREASE depth or η"
    elif DCR > 0.9:
        status = "warn"
        msg = f"TCC δ_total near L/240 limit (DCR={DCR:.3f})"
    else:
        status = "pass"
        msg = f"TCC δ_total={delta_total*1000:.1f}mm ≤ L/240 — OK"
    return CodeCheckResult(element_id, "tcc_deflection", "gravity", DCR, status, msg)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MEGA-TRUSS SIZING  (Chapter 6)
# ══════════════════════════════════════════════════════════════════════════════

def mega_truss_depth_m(span_m: float, heavy: bool = True) -> float:
    """
    Mega-truss depth: D = span / ratio.
    ratio = 8–15 per §6.1 (8 for transfer, 15 for lighter spans).
    """
    ratio = 8.0 if heavy else 12.0
    return span_m / ratio


def mega_truss_chord_force_kips(
    wu_kip_ft: float,
    span_ft: float,
    truss_depth_ft: float,
) -> tuple[float, float]:
    """
    Approximate chord forces in a simply supported mega-truss.
    Mu_max = wu·L²/8
    T = C = Mu / d  (d = truss depth)

    Returns (tension_chord_kips, compression_chord_kips).
    """
    Mu = wu_kip_ft * span_ft**2 / 8.0   # kip-ft
    chord_force = Mu / max(truss_depth_ft, 0.01)  # kips
    return chord_force, chord_force   # T = C for symmetric loading


def p_delta_amplification(
    P_kips: float,
    V_kips: float,
    delta_m: float,
    h_sx_m: float,
) -> float:
    """
    Second-order P-Δ stability coefficient θ (ASCE 7-22 §12.8.7).

    θ = P·Δ / (V·h_sx)

    If θ > 0.10: second-order analysis required.
    If θ > 0.25: system is unstable (design must be revised).

    Returns θ (dimensionless).
    """
    return (P_kips * delta_m * 3.2808) / max(V_kips * h_sx_m * 3.2808, 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SUMMARY / PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    vertices: np.ndarray,
    faces: np.ndarray,
    creases: np.ndarray,
    floor_height_m: float = 4.0,
    sdc: str = "D",
    program: str = "office",
    sdl_psf: float = 20.0,
) -> dict:
    """
    End-to-end pipeline:
    1. Extract mesh descriptors
    2. Select structural system
    3. Compute gravity loads
    4. Generate diagrid layout (if applicable)
    5. Return consolidated report dict

    Parameters
    ----------
    vertices       : (N, 3) vertex positions, metres
    faces          : (M, 3) triangle face indices
    creases        : (K, 2) crease edge indices
    floor_height_m : assumed floor-to-floor height, metres
    sdc            : seismic design category 'A'–'F'
    program        : occupancy type
    sdl_psf        : superimposed dead load, psf

    Returns
    -------
    dict with keys: 'descriptors', 'system', 'loads', 'diagrid', 'checks'
    """
    desc   = extract_mesh_descriptors(vertices, faces, creases, floor_height_m)
    system = select_structural_system(desc, sdc=sdc, program=program)
    loads  = gravity_loads_psf(program, sdl_psf)

    diagrid_layout = None
    if "DIAGRID" in system.lateral_system:
        diagrid_layout = generate_diagrid_layout(desc, floor_height_m)

    # Example code checks (requires actual analysis results to be wired in):
    checks = []

    return {
        "descriptors": desc,
        "system":      system,
        "loads":       loads,
        "diagrid":     diagrid_layout,
        "checks":      checks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE — NUMERIC DOMAIN TABLE
# (print for documentation / validation)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_TABLE = {
    # Label                          : (min, max, unit, code_ref)
    "Diagrid theta":                   (50.0,  75.0, "deg",   "Appendix C"),
    "Diagrid theta (balanced, default)":(62.0, 62.0, "deg",   "Ch.4 §4.2"),
    "Space frame depth/span":          (1/20,  1/10, "ratio", "Ch.5 §5.2"),
    "Space frame module":              (3.0,   6.0,  "m",     "Ch.5 §5.2"),
    "Truss depth/span (roof/floor)":   (1/18,  1/8,  "ratio", "Ch.6 §6.1"),
    "Mega-truss depth/span":           (1/15,  1/8,  "ratio", "Ch.6 §6.1"),
    "KL/r (compression)":             (0.0,  200.0, "—",     "AISC 360 §E2"),
    "Seismic drift (SDC D-F, office)":(0.0,  0.020, "Δ/h",   "ASCE 7-22 T12.12-1"),
    "Wind drift":                     (0.0, 1/400,  "H/n",   "AISC DG3"),
    "Beam LL deflection":             (0.0, 1/360,  "L/n",   "IBC 2024 T1604.3"),
    "Beam TL deflection":             (0.0, 1/240,  "L/n",   "IBC 2024 T1604.3"),
    "Roof LL deflection (>25ft span)":(0.0, 1/480,  "L/n",   "IBC 2024 T1604.3"),
    "Floor vibration fn":             (4.0,  None,  "Hz",    "AISC DG11"),
    "TCC floor vibration fn":         (8.0,  None,  "Hz",    "Ch.15 §15.3"),
    "RC wall min thickness":          (12.0, None,  "in",    "ACI 318 §18.10.2.3"),
    "RC long. rein. ratio ρl":        (0.0025, 0.006, "—",   "ACI 318 §18.10.2.1"),
    "Min composite ratio":            (0.25, 1.0,   "—",     "AISC 360 Commentary I3"),
    "Shell thin (λ = L/t)":          (20.0, None,  "—",     "Ch.17"),
    "Shell thick (λ = L/t)":         (None, 20.0,  "—",     "Ch.17"),
    "Shell 3D solid (λ = L/t)":      (None, 10.0,  "—",     "Ch.17"),
    "Boundary element trigger":        (None, 0.20, "σ/f'c", "ACI 318 §18.10.6"),
    "P-Δ check required (θ)":         (0.10, None, "—",     "ASCE 7-22 §12.8.7"),
    "P-Δ instability (θ)":            (0.25, None, "—",     "ASCE 7-22 §12.8.7"),
    "Void fraction → mega-truss":      (0.20, None, "fraction", "Ch.19"),
    "CFST H/D limit (circular)":      (None, 0.15, "Es/Fy", "AISC 360 §I2.2a"),
    "CFST b/t limit (rectangular)":   (None, 3.0,  "√(Es/Fy)", "AISC 360 §I2.2b"),
    "Min diagrid members at node":     (3,    None, "count", "Ch.20 §20.1"),
    "Stud spacing min":               (6.0,  None, "×dia",  "AISC 360 §I8.2d"),
    "Stud spacing max (longitudinal)":(None, 8.0,  "×t_slab","AISC 360 §I8.2d"),
    "Slab above deck min":            (3.5,  None, "in",    "AISC 360 §I3.2a"),
    "Fy A992 W-shapes":               (50.0, 50.0, "ksi",   "ASTM A992"),
    "Fy A500 Gr.C HSS":               (46.0, 46.0, "ksi",   "ASTM A500"),
    "E steel":                        (29000, 29000, "ksi",  "AISC"),
}


def print_domain_table():
    """Pretty-print all numeric domain limits."""
    print(f"\n{'─'*80}")
    print(f"{'PARAMETER':<42} {'MIN':>8} {'MAX':>8} {'UNIT':<10} CODE REF")
    print(f"{'─'*80}")
    for label, (lo, hi, unit, ref) in DOMAIN_TABLE.items():
        lo_s = f"{lo:.4g}" if lo is not None else "—"
        hi_s = f"{hi:.4g}" if hi is not None else "—"
        print(f"  {label:<40} {lo_s:>8} {hi_s:>8} {unit:<10} {ref}")
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    print_domain_table()

    # ── Quick smoke test with a synthetic box mesh ─────────────────────────
    import numpy as np

    # Simple box: 30m × 30m × 120m (H/W = 4, tall building)
    W, D, H = 30.0, 30.0, 120.0
    verts = np.array([
        [0, 0, 0],  [W, 0, 0],  [W, D, 0],  [0, D, 0],
        [0, 0, H],  [W, 0, H],  [W, D, H],  [0, D, H],
    ], dtype=float)
    fcs = np.array([
        [0,1,2],[0,2,3],   # bottom
        [4,5,6],[4,6,7],   # top
        [0,1,5],[0,5,4],   # front
        [1,2,6],[1,6,5],   # right
        [2,3,7],[2,7,6],   # back
        [3,0,4],[3,4,7],   # left
    ], dtype=int)
    creases = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]], dtype=int)

    result = run_full_pipeline(verts, fcs, creases, floor_height_m=4.0, sdc="D", program="office")
    d = result["descriptors"]
    s = result["system"]
    print(f"H/W = {d.H_W_ratio:.2f}  →  {d.aspect_category}")
    print(f"Lateral: {s.lateral_system}  |  Gravity: {s.gravity_system}")
    print(f"Stories: {d.story_count}  |  Outriggers: {s.n_outrigger_levels}")
    for note in s.notes:
        print(f"  • {note}")

    # Diagrid sizing
    theta = diagrid_theta_from_HW(d.H_W_ratio)
    print(f"\nDiagrid θ = {theta}°")
    Hm = diagrid_module_height_m(4.0, d.stories_per_module)
    perim = 2 * (d.width_m + d.depth_m)
    bays  = diagrid_bay_count(perim, Hm, theta)
    Ldiag = diagrid_diagonal_length_m(Hm, theta)
    print(f"Module height = {Hm:.1f}m  |  Bay count = {bays}  |  Diagonal L = {Ldiag:.2f}m")

    # Code check examples
    r1 = check_member_interaction("COL-001", Pr=500, Pc=800, Mrx=200, Mcx=400)
    r2 = check_slenderness("BRACE-001", KL_r=145)
    r3 = check_story_drift("STORY-10", delta_m=0.045, h_sx_m=4.0)
    for r in [r1, r2, r3]:
        print(f"[{r.status.upper():4}] {r.check_type}: {r.message}")
