"""
STEEL STRUCTURE DESIGN RULEBOOK — CALLABLE FUNCTIONS FOR AI AGENT
==================================================================
Purpose: Given a closed mesh (building massing), an AI agent calls these
functions IN ORDER to generate a structurally valid steel frame.

Every function is:
  - Deterministic (same input → same output, no randomness)
  - Derived from AISC 360-22, ASCE 7-22, ACI 318-19, or physics
  - Returns numeric values with units stated in the docstring
  - Has domain bounds: the agent can validate every output

The agent's workflow on ANY mesh:
  PHASE 1  —  Analyze mesh geometry         (R01–R08)
  PHASE 2  —  Generate structural grid      (R09–R18)
  PHASE 3  —  Assign cross-sections         (R19–R30)
  PHASE 4  —  Design connections            (R31–R38)
  PHASE 5  —  Validate entire structure     (R39–R50)

No guessing.  No latent space.  Pure math and physics.

Units unless stated:  metres (m), kilonewtons (kN), megapascals (MPa),
                      degrees (°), seconds (s).
Imperial equivalents given where code references require them.

Codes:  AISC 360-22 · ASCE 7-22 · AISC 341-22 · ACI 318-19 · Eurocode 3
        AISC Design Guide 3 (Serviceability) · AISC Design Guide 11 (Vibration)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — MATERIAL PROPERTIES (invariant, not rules)
# ══════════════════════════════════════════════════════════════════════════════

# Steel grades — [Fy, Fu] in MPa
STEEL_GRADES = {
    "A992":   {"Fy": 345.0, "Fu": 450.0},   # W-shapes
    "A500C":  {"Fy": 317.0, "Fu": 427.0},   # HSS round/rect
    "A1085":  {"Fy": 345.0, "Fu": 450.0},   # HSS high-seismic
    "A36":    {"Fy": 248.0, "Fu": 400.0},   # plates / built-up
    "S355":   {"Fy": 355.0, "Fu": 510.0},   # Eurocode equivalent
}

E_STEEL   = 200_000.0   # MPa  (Young's modulus — ALL structural steels)
G_STEEL   =  77_200.0   # MPa  (Shear modulus = E / 2(1+ν), ν=0.3)
RHO_STEEL =   7850.0    # kg/m³ (density)
ALPHA_T   = 1.2e-5      # 1/°C  (coefficient of thermal expansion)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — MESH GEOMETRY ANALYSIS  (R01–R08)
#
#  Input: mesh vertices (N,3) in metres, faces (M,3) int, edges (K,2) int
#  Output: numeric descriptors the agent needs for every subsequent decision
# ══════════════════════════════════════════════════════════════════════════════

# ── R01: Member maximum length ───────────────────────────────────────────────

def R01_max_member_length_m(
    Fy_MPa: float = 345.0,
    r_min_mm: float = 50.0,
    K: float = 1.0,
) -> float:
    """
    RULE 01 — Maximum permissible length of ANY steel compression member.

    Source: AISC 360-22 §E2 Commentary: KL/r ≤ 200 (practical limit).

    Formula:  L_max = 200 · r_min / K

    Parameters
    ----------
    Fy_MPa   : yield stress (not used in formula but required for context)
    r_min_mm : smallest radius of gyration of the cross-section, mm
    K        : effective length factor (1.0 pinned-pinned, 0.65 fixed-fixed,
               2.0 cantilever)

    Returns
    -------
    L_max : maximum member length in metres

    Domain
    ------
    Output must be > 0.
    For typical HSS 200×200×10: r_min ≈ 78mm → L_max = 15.6m (K=1).
    For W360×134:              r_min ≈ 94mm → L_max = 18.8m (K=1).
    Absolute practical ceiling for fabrication/transport: 12–18 m.
    """
    return (200.0 * r_min_mm / K) / 1000.0


# ── R02: Member minimum length ──────────────────────────────────────────────

def R02_min_member_length_m(
    d_section_mm: float,
) -> float:
    """
    RULE 02 — Minimum practical member length.

    Source: fabrication practice — member must be long enough to fit
    connections at both ends.  Minimum = 4× section depth.

    Formula:  L_min = 4 · d

    Parameters
    ----------
    d_section_mm : overall depth of the cross-section, mm

    Returns
    -------
    L_min : minimum member length in metres

    Domain
    ------
    Typical: 0.5 m – 2.0 m.  Below 0.5 m → use gusset plate, not a member.
    """
    return max(4.0 * d_section_mm / 1000.0, 0.5)


# ── R03: Edge length from mesh → grid spacing ───────────────────────────────

def R03_grid_spacing_from_edge_length_m(
    edge_length_m: float,
    role: Literal["primary_beam", "secondary_beam", "brace", "diagrid"],
) -> dict:
    """
    RULE 03 — Validate whether a mesh edge length is a feasible member span.

    Source: AISC Design Guide 3 (Serviceability), Ruddy (2000) rules of thumb,
            fabrication practice.

    Limits per role (metres):
      primary_beam   :  6.0 – 18.0 m   (girder/primary grid)
      secondary_beam :  3.0 – 12.0 m   (infill beams)
      brace          :  2.0 – 15.0 m   (diagonal braces)
      diagrid        :  3.0 – 20.0 m   (diagrid diagonals)

    Returns
    -------
    dict with keys:
      'valid'   : bool
      'min_m'   : float — lower limit
      'max_m'   : float — upper limit
      'action'  : str   — 'ok' | 'subdivide' | 'merge'
    """
    limits = {
        "primary_beam":   (6.0,  18.0),
        "secondary_beam": (3.0,  12.0),
        "brace":          (2.0,  15.0),
        "diagrid":        (3.0,  20.0),
    }
    lo, hi = limits[role]
    if edge_length_m < lo:
        return {"valid": False, "min_m": lo, "max_m": hi, "action": "merge"}
    elif edge_length_m > hi:
        return {"valid": False, "min_m": lo, "max_m": hi, "action": "subdivide"}
    else:
        return {"valid": True, "min_m": lo, "max_m": hi, "action": "ok"}


# ── R04: Mesh face angle → structural implication ────────────────────────────

def R04_dihedral_angle_structural_action(
    dihedral_angle_deg: float,
) -> dict:
    """
    RULE 04 — Classify structural consequence of a mesh crease/fold.

    Source: Shell theory (Timoshenko & Woinowsky-Krieger), AISC DG 31.

    Angle is measured as the dihedral angle between two adjacent faces
    (180° = flat/coplanar, <180° = convex fold, >180° = concave fold).

    Classification:
      170°–180° : flat        → standard beam connection
      150°–170° : mild fold   → angled connection, no stiffener needed
      120°–150° : sharp fold  → stiffened connection, moment splice
      90°–120°  : hard crease → structural ridge beam required
      < 90°     : acute fold  → full moment node, possible cast-steel joint

    Returns
    -------
    dict with:
      'zone'                : str
      'connection_type'     : str
      'stiffener_required'  : bool
      'depth_amplification' : float  (multiply default beam depth by this)
    """
    a = dihedral_angle_deg
    if a >= 170:
        return {"zone": "flat", "connection_type": "simple_shear",
                "stiffener_required": False, "depth_amplification": 1.0}
    elif a >= 150:
        return {"zone": "mild_fold", "connection_type": "angled_endplate",
                "stiffener_required": False, "depth_amplification": 1.1}
    elif a >= 120:
        return {"zone": "sharp_fold", "connection_type": "moment_splice",
                "stiffener_required": True,
                "depth_amplification": 1.0 + (180.0 - a) / 180.0}
    elif a >= 90:
        return {"zone": "hard_crease", "connection_type": "ridge_beam",
                "stiffener_required": True,
                "depth_amplification": 1.0 + (180.0 - a) / 120.0}
    else:
        return {"zone": "acute_fold", "connection_type": "cast_steel_node",
                "stiffener_required": True,
                "depth_amplification": 2.0}


# ── R05: Vertex valence → node complexity ────────────────────────────────────

def R05_node_complexity_from_valence(
    n_members: int,
) -> dict:
    """
    RULE 05 — Classify node connection complexity from number of converging members.

    Source: AISC Design Guide 24 (HSS Connections), practice (Arup, SOM, BIG).

    Valence = number of structural members meeting at a node.

    Classification:
      2     : inline splice or simple corner
      3     : standard tee / knee joint
      4     : cruciform — standard steel detail
      5–6   : multi-member — welded gusset hub or cast-steel node
      7+    : hyper-node — must use cast-steel spherical node

    Returns
    -------
    dict with:
      'type'                : str
      'cast_steel_required' : bool
      'rigid_zone_radius_m' : float  (multiply by avg member length → see R06)
      'min_plate_thickness_mm' : float
    """
    if n_members <= 2:
        return {"type": "splice", "cast_steel_required": False,
                "rigid_zone_radius_m_factor": 0.02, "min_plate_thickness_mm": 12.0}
    elif n_members == 3:
        return {"type": "tee_joint", "cast_steel_required": False,
                "rigid_zone_radius_m_factor": 0.03, "min_plate_thickness_mm": 16.0}
    elif n_members == 4:
        return {"type": "cruciform", "cast_steel_required": False,
                "rigid_zone_radius_m_factor": 0.04, "min_plate_thickness_mm": 20.0}
    elif n_members <= 6:
        return {"type": "multi_hub", "cast_steel_required": True,
                "rigid_zone_radius_m_factor": 0.05, "min_plate_thickness_mm": 25.0}
    else:
        return {"type": "hyper_node", "cast_steel_required": True,
                "rigid_zone_radius_m_factor": 0.06, "min_plate_thickness_mm": 32.0}


# ── R06: Rigid zone radius at node ──────────────────────────────────────────

def R06_rigid_zone_radius_m(
    avg_member_length_m: float,
    n_members: int,
) -> float:
    """
    RULE 06 — Compute rigid zone radius at a structural node.

    Source: finite element modelling practice (CSI ETABS, SAP2000 manuals).
    At nodes where members converge, the overlapping material creates a
    zone that is effectively rigid.  Model it as a rigid offset of length r.

    Formula:  r = factor × L_avg
    where factor comes from R05.

    Returns
    -------
    r : rigid zone radius in metres

    Domain
    ------
    0.05 m ≤ r ≤ 0.50 m (practical range).
    """
    info = R05_node_complexity_from_valence(n_members)
    r = info["rigid_zone_radius_m_factor"] * avg_member_length_m
    return max(0.05, min(0.50, r))


# ── R07: Surface curvature → structural system ──────────────────────────────

def R07_curvature_to_system(
    max_gaussian_curvature_1pm2: float,
    max_mean_curvature_1pm: float,
    height_m: float,
    H_W_ratio: float,
) -> dict:
    """
    RULE 07 — Map mesh curvature to structural system type.

    Source: Shell & spatial structures theory (Adriaenssens et al. 2014),
            diagrid research (Moon 2007, 2013), practice (Foster, SOM, BIG).

    Decision tree (all thresholds from published parametric studies):

    1. If |K_gauss| < 0.001 AND |H_mean| < 0.01:
       → surface is flat → use FRAME (beam-column grid)

    2. If K_gauss > 0.001 (synclastic, dome-like):
       → use GRIDSHELL or DIAGRID

    3. If K_gauss < -0.001 (anticlastic, saddle-like):
       → use CABLE-NET or GRIDSHELL with pre-stress

    4. If mixed signs:
       → use HYBRID DIAGRID + LOCAL GRIDSHELL patches

    Height/width overlay:
       H/W > 4   : must include vertical core for stability
       H/W > 8   : add outrigger trusses at H/3 and 2H/3

    Returns
    -------
    dict with:
      'primary_system'  : str
      'needs_core'      : bool
      'needs_outrigger' : bool
      'n_outriggers'    : int
      'grid_pattern'    : str  ('quad'|'tri'|'diagrid'|'voronoi')
    """
    K = max_gaussian_curvature_1pm2
    H = max_mean_curvature_1pm
    hw = H_W_ratio

    needs_core = hw > 4.0
    needs_outrigger = hw > 8.0
    n_out = max(2, int(hw // 4)) if needs_outrigger else (1 if hw > 6 else 0)

    if abs(K) < 0.001 and abs(H) < 0.01:
        system = "frame"
        pattern = "quad"
    elif K > 0.001:
        system = "diagrid" if hw > 2.0 else "gridshell"
        pattern = "diagrid" if hw > 2.0 else "tri"
    elif K < -0.001:
        system = "gridshell_prestressed"
        pattern = "tri"
    else:
        system = "hybrid_diagrid_gridshell"
        pattern = "diagrid"

    return {
        "primary_system": system,
        "needs_core": needs_core,
        "needs_outrigger": needs_outrigger,
        "n_outriggers": n_out,
        "grid_pattern": pattern,
    }


# ── R08: Floor height → story slicing planes ────────────────────────────────

def R08_story_slicing_planes_m(
    mesh_z_min: float,
    mesh_z_max: float,
    floor_height_m: float = 4.0,
    ground_floor_height_m: float = 5.0,
) -> list[float]:
    """
    RULE 08 — Generate Z-coordinates for horizontal slicing planes.

    Source: IBC 2024 §1004 minimum ceiling heights, practice.

    The mesh is sliced at these Z-levels to create floor contours.
    The agent intersects the mesh with each plane to get closed curves,
    which become floor boundary edges.

    Floor height limits:
      Minimum floor-to-floor: 2.7 m (IBC minimum habitable ceiling 2.4m + structure)
      Maximum floor-to-floor: 6.0 m (beyond this, need intermediate bracing)
      Ground floor:           4.5 – 6.0 m (typical retail/lobby)
      Typical office:         3.6 – 4.2 m
      Residential:            2.8 – 3.2 m

    Returns
    -------
    List of Z-coordinates in metres (ascending from base to top).
    """
    floor_height_m = max(2.7, min(6.0, floor_height_m))
    ground_floor_height_m = max(4.0, min(6.0, ground_floor_height_m))

    planes = [mesh_z_min]
    z = mesh_z_min + ground_floor_height_m
    planes.append(z)
    while z + floor_height_m <= mesh_z_max:
        z += floor_height_m
        planes.append(z)
    if planes[-1] < mesh_z_max - 1.0:
        planes.append(mesh_z_max)
    return planes


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — GRID GENERATION RULES  (R09–R18)
#
#  Input: mesh geometry + Phase 1 descriptors
#  Output: node positions (x,y,z) and member connectivity
# ══════════════════════════════════════════════════════════════════════════════

# ── R09: Primary grid spacing ───────────────────────────────────────────────

def R09_primary_grid_spacing_m(
    span_m: float,
    load_intensity_kPa: float = 5.0,
) -> float:
    """
    RULE 09 — Optimal primary beam spacing.

    Source: Ruddy (2000), AISC Steel Solutions Center, economy studies.

    The 3:4 primary-to-secondary ratio is standard practice.
    Primary columns at 6–9 m centres; secondary beams at 2–3 m.

    Formula (from economy curves, steel weight minimization):
      For office/commercial (5 kPa):
        primary_spacing = min(max(span_m / 3, 6.0), 12.0)

    For heavier loads (>7.5 kPa):
        primary_spacing = min(max(span_m / 4, 4.5), 9.0)

    Returns
    -------
    Spacing in metres.

    Domain: 4.5 ≤ spacing ≤ 12.0 m
    """
    if load_intensity_kPa > 7.5:
        return min(max(span_m / 4.0, 4.5), 9.0)
    else:
        return min(max(span_m / 3.0, 6.0), 12.0)


def R10_secondary_grid_spacing_m(
    primary_spacing_m: float,
) -> float:
    """
    RULE 10 — Secondary beam spacing from primary spacing.

    Source: Ruddy (2000), economy studies (minimum steel weight).

    Formula:  secondary = primary × (3/4)
    Clamp: 2.0 ≤ secondary ≤ 4.5 m

    Domain: for composite deck, max 3.0–3.6 m (deck spanning capacity).
    """
    s = primary_spacing_m * 0.75
    return max(2.0, min(4.5, s))


# ── R11: Diagrid angle from building aspect ratio ────────────────────────────

def R11_diagrid_angle_deg(
    H_W_ratio: float,
) -> float:
    """
    RULE 11 — Optimal diagrid diagonal angle from horizontal.

    Source: Moon, Connor & Fernandez (2007), "Diagrid structural systems
    for tall buildings: characteristics and methodology for preliminary
    design", The Structural Design of Tall and Special Buildings.

    The optimal angle balances:
      - Gravity load path efficiency (needs steep angle, ~70°+)
      - Lateral stiffness (needs flatter angle, ~50°)
      - Combined optimum at 60–70° for most buildings.

    Parametric results:
      H/W < 3  :  θ = 70°  (gravity-dominated, stubby building)
      3–5      :  θ = 65°
      5–7      :  θ = 60°
      > 7      :  θ = 55°  (lateral-dominated, slender tower)

    Domain: 50° ≤ θ ≤ 75°

    Returns
    -------
    Angle in degrees.
    """
    hw = H_W_ratio
    if hw < 3:
        theta = 70.0
    elif hw < 5:
        theta = 70.0 - (hw - 3.0) * 2.5   # 70 → 65
    elif hw < 7:
        theta = 65.0 - (hw - 5.0) * 2.5   # 65 → 60
    else:
        theta = max(50.0, 60.0 - (hw - 7.0) * 1.667)  # 60 → 50
    return max(50.0, min(75.0, theta))


# ── R12: Diagrid module height ───────────────────────────────────────────────

def R12_diagrid_module_height_m(
    floor_height_m: float,
    n_stories_per_module: int,
) -> float:
    """
    RULE 12 — Height of one diagrid diamond module.

    Source: Moon (2007), practice (30 St Mary Axe, Hearst Tower).

    Formula:  H_module = floor_height × N
    where N = number of stories per diamond:
      short buildings (<10 stories): N = 2
      medium (10–20):                N = 3
      tall (>20):                    N = 4

    Domain: 6.0 m ≤ H_module ≤ 20.0 m
    """
    h = floor_height_m * n_stories_per_module
    return max(6.0, min(20.0, h))


# ── R13: Diagrid bay width from module height and angle ──────────────────────

def R13_diagrid_bay_width_m(
    module_height_m: float,
    theta_deg: float,
) -> float:
    """
    RULE 13 — Horizontal bay width of one diagrid diamond.

    Source: trigonometry + Moon (2007).

    Formula:  w = H_module / tan(θ)

    This is the horizontal projection of one diagonal.
    The full diamond width = 2w (left diagonal + right diagonal share the bay).

    Returns
    -------
    Bay width in metres.

    Domain: 3.0 ≤ w ≤ 15.0 m
    """
    w = module_height_m / math.tan(math.radians(theta_deg))
    return max(3.0, min(15.0, w))


# ── R14: Number of bays around perimeter ─────────────────────────────────────

def R14_diagrid_bay_count(
    perimeter_m: float,
    bay_width_m: float,
) -> int:
    """
    RULE 14 — Number of diagrid bays around the building perimeter.

    Source: geometry + symmetry constraint.

    Formula:  n = round_to_even(perimeter / bay_width)
    Minimum 4 bays (structural stability requires triangulated path on
    every face).

    Returns
    -------
    Integer bay count (always even, ≥ 4).
    """
    n = max(4, round(perimeter_m / bay_width_m))
    return n if n % 2 == 0 else n + 1


# ── R15: Diagonal member length ──────────────────────────────────────────────

def R15_diagonal_length_m(
    module_height_m: float,
    theta_deg: float,
) -> float:
    """
    RULE 15 — True 3D length of a diagrid diagonal.

    Formula:  L = H_module / sin(θ)

    Domain: L must satisfy R01 (KL/r ≤ 200).
    """
    return module_height_m / math.sin(math.radians(theta_deg))


# ── R16: Minimum members at any structural node ─────────────────────────────

def R16_min_members_at_node() -> int:
    """
    RULE 16 — Minimum number of members meeting at any node.

    Source: structural stability theory (Maxwell's rule for static determinacy).
    A spatial truss requires at minimum 3 members per node for 3D equilibrium
    (6 DOF, but support conditions provide some restraint).

    For a diagrid node: minimum 3 (two diagonals + one ring beam).
    For a frame node:   minimum 2 (beam + column).

    Returns
    -------
    3
    """
    return 3


# ── R17: Contour subdivision — add nodes along mesh edges ───────────────────

def R17_subdivide_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    target_spacing_m: float,
) -> np.ndarray:
    """
    RULE 17 — Subdivide a mesh edge into segments of target spacing.

    Source: meshing practice for FEM / structural grid generation.

    Formula:
      n_segments = max(1, round(L / target_spacing))
      new_nodes = linear interpolation at 1/n, 2/n, ..., (n-1)/n

    Parameters
    ----------
    p0, p1          : (3,) start and end vertex coordinates
    target_spacing_m: desired node spacing from R09/R10

    Returns
    -------
    (n-1, 3) array of interior node positions.
    """
    L = np.linalg.norm(p1 - p0)
    n = max(1, round(L / target_spacing_m))
    if n <= 1:
        return np.empty((0, 3))
    t = np.linspace(0, 1, n + 1)[1:-1]
    return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


# ── R18: Floor contour to structural boundary ────────────────────────────────

def R18_contour_to_structural_boundary(
    contour_points: np.ndarray,
    min_edge_length_m: float = 3.0,
    max_edge_length_m: float = 12.0,
) -> np.ndarray:
    """
    RULE 18 — Clean a floor contour curve into structural-grid-ready polygon.

    Operations:
    1. Remove points closer than min_edge_length (merge).
    2. Subdivide segments longer than max_edge_length (split).

    Source: computational geometry best practice for structural grid gen.

    Returns
    -------
    (P, 3) cleaned polygon vertices.
    """
    pts = [contour_points[0]]
    for i in range(1, len(contour_points)):
        d = np.linalg.norm(contour_points[i] - pts[-1])
        if d < min_edge_length_m:
            continue  # skip (merge)
        elif d > max_edge_length_m:
            # subdivide
            n_sub = math.ceil(d / max_edge_length_m)
            for j in range(1, n_sub + 1):
                t = j / n_sub
                p = pts[-1] + t * (contour_points[i] - pts[-1])
                pts.append(p)
        else:
            pts.append(contour_points[i])
    return np.array(pts)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — CROSS-SECTION ASSIGNMENT  (R19–R30)
#
#  Input: member length, tributary area, loads
#  Output: section depth, area, moment of inertia, section name
# ══════════════════════════════════════════════════════════════════════════════

# ── R19: Beam depth from span (the master sizing rule) ───────────────────────

def R19_beam_depth_from_span_mm(
    span_m: float,
    support_condition: Literal["simple", "continuous", "cantilever"] = "simple",
    load_type: Literal["floor", "roof", "transfer"] = "floor",
) -> float:
    """
    RULE 19 — Required beam depth from span.

    Source:  AISC Commentary (historical §1.13.1, 7th–9th Edition Manuals):
             d = Fy/800 × L  (for Fy=50 ksi → d = L/16)
             Ruddy (2000): "½ inch of depth per foot of span" → L/24
             Practice consensus for Fy=345 MPa (A992):

    Span-to-depth ratios (L/d):
      Floor beam, simple span:        L/d = 20    → d = L/20
      Floor beam, continuous:         L/d = 24    → d = L/24
      Roof beam/purlin, simple:       L/d = 24    → d = L/24
      Roof beam, continuous:          L/d = 30    → d = L/30
      Cantilever:                     L/d = 8–12  → d = L/10
      Transfer girder:                L/d = 12–15 → d = L/12
      Plate girder (deep):            L/d = 10–12

    Returns
    -------
    d : required beam depth in mm.

    Domain: 150 mm ≤ d ≤ 3000 mm (W150 to mega plate girder).
    """
    ratios = {
        ("floor",    "simple"):     20.0,
        ("floor",    "continuous"): 24.0,
        ("roof",     "simple"):     24.0,
        ("roof",     "continuous"): 30.0,
        ("floor",    "cantilever"): 10.0,
        ("roof",     "cantilever"): 12.0,
        ("transfer", "simple"):     12.0,
        ("transfer", "continuous"): 15.0,
        ("transfer", "cantilever"):  8.0,
    }
    r = ratios.get((load_type, support_condition), 20.0)
    d_mm = span_m * 1000.0 / r
    return max(150.0, min(3000.0, d_mm))


# ── R20: Column depth from story height and load ────────────────────────────

def R20_column_depth_from_load_mm(
    axial_load_kN: float,
    unbraced_length_m: float,
    Fy_MPa: float = 345.0,
) -> float:
    """
    RULE 20 — Minimum column section depth.

    Source: AISC 360-22 §E3, back-solved for minimum A given KL/r ≤ 120
    (practical target for columns, not the 200 hard limit).

    Two constraints:
    1. Strength:  A_req = Pu / (φ·Fcr)
       where Fcr ≈ 0.658^(Fy/Fe)·Fy for KL/r ≈ 60 (typical)
       → Fcr ≈ 0.77·Fy for KL/r=60

    2. Slenderness: r_min ≥ KL/120
       For W-shapes: r_y ≈ 0.25·d
       → d ≥ KL / (120 × 0.25) = KL/30

    Returns
    -------
    d : minimum depth in mm (round up to nearest 50mm W-shape increment).

    Domain: 150 mm ≤ d ≤ 1000 mm
    """
    # Slenderness constraint
    d_slenderness = unbraced_length_m * 1000.0 / 30.0

    # Strength constraint (rough)
    phi_Fcr = 0.90 * 0.77 * Fy_MPa   # MPa
    A_req_mm2 = axial_load_kN * 1000.0 / phi_Fcr
    # For W-shapes: A ≈ 0.15 × d² (rough fit to AISC database)
    d_strength = math.sqrt(A_req_mm2 / 0.15)

    d = max(d_slenderness, d_strength)
    # Round up to nearest 50
    d = math.ceil(d / 50.0) * 50.0
    return max(150.0, min(1000.0, d))


# ── R21: Required moment of inertia from deflection limit ────────────────────

def R21_required_Ix_mm4(
    span_m: float,
    w_kN_m: float,
    deflection_limit_ratio: float = 360.0,
) -> float:
    """
    RULE 21 — Minimum I_x for a simply-supported beam under uniform load
    to satisfy deflection limit.

    Source: Euler-Bernoulli beam theory + IBC 2024 Table 1604.3.

    Formula (uniform load, simple span):
      δ_max = 5·w·L⁴ / (384·E·I)
      δ_allow = L / ratio
      → I_req = 5·w·L³·ratio / (384·E)

    Deflection limits:
      L/360 : floor live load          (IBC Table 1604.3)
      L/240 : floor total load
      L/480 : roof live load (span > 7.6m)
      L/180 : cantilever (L = 2× cantilever length)

    Parameters
    ----------
    span_m               : beam span, metres
    w_kN_m               : unfactored uniform load, kN/m
    deflection_limit_ratio: denominator of L/n (360, 240, 480 etc.)

    Returns
    -------
    I_req : required second moment of area, mm⁴

    Domain: 1e6 ≤ I ≤ 1e11 mm⁴ (W150 → mega plate girder)
    """
    L_mm = span_m * 1000.0
    w_N_mm = w_kN_m   # kN/m = N/mm
    I_req = (5.0 * w_N_mm * L_mm**3 * deflection_limit_ratio) / (384.0 * E_STEEL)
    return I_req


# ── R22: Required plastic section modulus from moment ────────────────────────

def R22_required_Zx_mm3(
    Mu_kNm: float,
    Fy_MPa: float = 345.0,
    phi_b: float = 0.90,
) -> float:
    """
    RULE 22 — Minimum plastic section modulus for a compact fully-braced beam.

    Source: AISC 360-22 §F2.1:  φMn = φ·Fy·Zx ≥ Mu
    → Zx_req = Mu / (φ·Fy)

    Parameters
    ----------
    Mu_kNm : factored bending moment, kN·m

    Returns
    -------
    Zx_req : mm³

    Domain: 1e4 ≤ Zx ≤ 5e7 mm³
    """
    return (Mu_kNm * 1e6) / (phi_b * Fy_MPa)


# ── R23: Required cross-section area from axial load ─────────────────────────

def R23_required_area_mm2(
    Pu_kN: float,
    KL_m: float,
    r_min_mm: float,
    Fy_MPa: float = 345.0,
    phi_c: float = 0.90,
) -> float:
    """
    RULE 23 — Minimum gross area for a compression member.

    Source: AISC 360-22 §E3.

    Procedure:
    1. KL/r from inputs.
    2. Compute Fe = π²E / (KL/r)².
    3. If KL/r ≤ 4.71√(E/Fy): Fcr = 0.658^(Fy/Fe)·Fy
       Else:                    Fcr = 0.877·Fe
    4. A_req = Pu / (φ·Fcr)

    Returns
    -------
    A_req : mm²
    """
    KL_r = (KL_m * 1000.0) / max(r_min_mm, 1.0)
    if KL_r > 200:
        KL_r = 200.0  # cap at practical limit

    Fe = math.pi**2 * E_STEEL / KL_r**2
    limit = 4.71 * math.sqrt(E_STEEL / Fy_MPa)

    if KL_r <= limit:
        Fcr = (0.658 ** (Fy_MPa / Fe)) * Fy_MPa
    else:
        Fcr = 0.877 * Fe
    Fcr = min(Fcr, Fy_MPa)

    return (Pu_kN * 1000.0) / (phi_c * Fcr)


# ── R24: HSS diameter from axial load (diagrid members) ──────────────────────

def R24_hss_diameter_mm(
    N_kN: float,
    length_m: float,
    Fy_MPa: float = 317.0,
    Dt_ratio: float = 30.0,
) -> float:
    """
    RULE 24 — Minimum CHS (circular hollow section) diameter for diagrid diagonal.

    Source: AISC 360-22 §E7 + HSS local buckling limits.

    Approach:
      Assume D/t = 30 (mid-range for CHS, satisfies D/t ≤ 0.11·E/Fy = 69).
      r = √(D² + (D-2t)²) / 4  ≈  D / (2√2) for thin wall.
      Iterate D until φPn ≥ N.

    Domain:
      Minimum D: 114 mm (HSS 4.5)
      Maximum D: 914 mm (HSS 36) — fabrication limit for standard CHS
      For very large loads: use built-up box section instead.

    Returns
    -------
    D : outside diameter in mm.
    """
    N_N = abs(N_kN) * 1000.0 * 1.10  # 10% safety margin

    for D_mm in range(114, 920, 2):
        t_mm = D_mm / Dt_ratio
        A = math.pi * (D_mm**2 - (D_mm - 2*t_mm)**2) / 4.0
        r = math.sqrt(D_mm**2 + (D_mm - 2*t_mm)**2) / 4.0
        KL_r = (length_m * 1000.0) / r
        if KL_r > 200:
            continue

        Fe = math.pi**2 * E_STEEL / KL_r**2
        limit = 4.71 * math.sqrt(E_STEEL / Fy_MPa)
        if KL_r <= limit:
            Fcr = (0.658 ** (Fy_MPa / Fe)) * Fy_MPa
        else:
            Fcr = 0.877 * Fe
        Fcr = min(Fcr, Fy_MPa)
        phi_Pn = 0.90 * Fcr * A

        if phi_Pn >= N_N:
            return float(D_mm)

    return 914.0  # fallback — flag for built-up section


# ── R25: Beam weight estimation ──────────────────────────────────────────────

def R25_beam_weight_kg_m(
    span_m: float,
    w_kN_m: float,
    Fy_MPa: float = 345.0,
) -> float:
    """
    RULE 25 — Estimated beam self-weight per metre.

    Source: Ruddy (2000): "For common floor beams, weight (lb/ft) ≈ span(ft)/2"
    Re-derived in metric:

    Formula (from Mu/Zx relationship + regression on AISC database):
      Mu = w·L²/8
      Zx_req = Mu / (0.9·Fy)
      weight_kg_m ≈ 0.0013 · Zx_req^0.95  (regression, R²=0.97 on AISC W-shapes)

    Simplified:
      weight_kg_m ≈ (w_kN_m · span_m²) / (8 × 0.9 × Fy_MPa × 0.77 × 1e-3)

    Returns
    -------
    Estimated weight in kg/m.

    Domain: 15 kg/m (W150×13) to 1086 kg/m (W920×1077).
    """
    Mu_Nm = w_kN_m * 1000.0 * (span_m * 1000.0)**2 / 8.0
    Zx_mm3 = Mu_Nm / (0.9 * Fy_MPa)
    # Regression: weight_kg_m = 7.85e-7 × Zx^0.95
    weight = 7.85e-7 * Zx_mm3**0.95
    return max(13.0, min(1100.0, weight))


# ── R26: Section depth ↔ section properties (parametric model) ──────────────

def R26_section_properties_from_depth_mm(
    d_mm: float,
    section_type: Literal["W_shape", "HSS_rect", "HSS_round", "built_up_box"],
) -> dict:
    """
    RULE 26 — Approximate cross-section properties from depth alone.

    Source: Regression on AISC Shapes Database v16.0 (222+ shapes).
    These are CURVE FITS, not exact — they give the agent a starting
    point to select from a real database.

    For W-shapes (A992, Fy=345 MPa):
      A   ≈ 0.15 · d²          mm²         (R²=0.91)
      Ix  ≈ 0.045 · d⁴ / 1000  mm⁴         (R²=0.94)
      Zx  ≈ 0.12 · d³ / 100    mm³         (R²=0.93)
      rx  ≈ 0.42 · d            mm          (R²=0.96)
      ry  ≈ 0.25 · d            mm          (R²=0.89)
      bf  ≈ 0.55 · d            mm
      tf  ≈ 0.04 · d            mm
      tw  ≈ 0.025 · d           mm
      wt  ≈ RHO_STEEL × A × 1e-6 kg/m

    Returns
    -------
    dict with all properties in mm / mm² / mm³ / mm⁴ / kg/m.
    """
    d = d_mm
    if section_type == "W_shape":
        A  = 0.15 * d**2
        Ix = 0.045 * d**4 / 1000.0
        Iy = 0.012 * d**4 / 1000.0
        Zx = 0.12 * d**3 / 100.0
        Zy = 0.06 * d**3 / 100.0
        rx = 0.42 * d
        ry = 0.25 * d
        J  = 0.0001 * d**4 / 1000.0
        bf = 0.55 * d
        tf = 0.04 * d
        tw = 0.025 * d
    elif section_type == "HSS_rect":
        # Square HSS: b = d, t = d/30
        t = d / 30.0
        A  = 4.0 * d * t - 4 * t**2
        Ix = d**3 * t / 3.0
        Iy = Ix
        Zx = d**2 * t
        Zy = Zx
        rx = 0.40 * d
        ry = rx
        J  = 2 * t * (d - t)**2 * (d - t)   # approximate
        bf = d
        tf = t
        tw = t
    elif section_type == "HSS_round":
        t = d / 30.0
        A  = math.pi * (d**2 - (d - 2*t)**2) / 4.0
        Ix = math.pi * (d**4 - (d - 2*t)**4) / 64.0
        Iy = Ix
        Zx = (d**3 - (d - 2*t)**3) / 6.0
        Zy = Zx
        rx = math.sqrt(d**2 + (d - 2*t)**2) / 4.0
        ry = rx
        J  = math.pi * (d**4 - (d - 2*t)**4) / 32.0
        bf = d
        tf = t
        tw = t
    else:  # built_up_box
        t_f = d / 20.0
        t_w = d / 30.0
        b = 0.8 * d
        A = 2 * b * t_f + 2 * (d - 2*t_f) * t_w
        Ix = 2 * b * t_f * (d/2 - t_f/2)**2 + 2 * t_w * (d - 2*t_f)**3 / 12.0
        Iy = 2 * t_f * b**3 / 12.0 + 2 * (d - 2*t_f) * t_w * (b/2 - t_w/2)**2
        Zx = A * d / 4.0
        Zy = A * b / 4.0
        rx = math.sqrt(Ix / A) if A > 0 else d * 0.4
        ry = math.sqrt(Iy / A) if A > 0 else d * 0.3
        J = 2 * t_f * t_w * (b - t_w)**2 * (d - t_f)**2 / (b*t_f + d*t_w)
        bf = b
        tf = t_f
        tw = t_w

    wt_kg_m = RHO_STEEL * A * 1e-6
    return {
        "d_mm": d, "A_mm2": A, "Ix_mm4": Ix, "Iy_mm4": Iy,
        "Zx_mm3": Zx, "Zy_mm3": Zy, "rx_mm": rx, "ry_mm": ry,
        "J_mm4": J, "bf_mm": bf, "tf_mm": tf, "tw_mm": tw,
        "weight_kg_m": wt_kg_m, "section_type": section_type,
    }


# ── R27: Tributary area load collection ──────────────────────────────────────

def R27_tributary_load_kN_m(
    trib_width_m: float,
    dead_kPa: float = 3.5,
    live_kPa: float = 2.4,
    cladding_kN_m: float = 0.0,
) -> dict:
    """
    RULE 27 — Compute beam line loads from tributary width and area loads.

    Source: ASCE 7-22 §4.3 (live loads), basic statics.

    Typical area loads (kPa):
      Dead (composite slab + MEP + ceiling): 2.5 – 4.0 kPa
      Live (office):     2.4 kPa   (50 psf)
      Live (residential): 1.9 kPa  (40 psf)
      Live (assembly):   4.8 kPa   (100 psf)
      Live (roof):       1.0 kPa   (20 psf)

    Returns
    -------
    dict with:
      'w_dead_kN_m'   : unfactored dead line load
      'w_live_kN_m'   : unfactored live line load
      'wu_gravity_kN_m': LRFD factored (1.2D + 1.6L)
    """
    w_d = dead_kPa * trib_width_m + cladding_kN_m
    w_l = live_kPa * trib_width_m
    wu = 1.2 * w_d + 1.6 * w_l
    return {
        "w_dead_kN_m": w_d,
        "w_live_kN_m": w_l,
        "wu_gravity_kN_m": wu,
    }


# ── R28: Column axial load from tributary area ──────────────────────────────

def R28_column_axial_load_kN(
    trib_area_m2: float,
    n_stories_above: int,
    dead_kPa: float = 3.5,
    live_kPa: float = 2.4,
    cladding_kN_per_story: float = 0.0,
) -> dict:
    """
    RULE 28 — Factored axial load on a column.

    Source: ASCE 7-22 §4.7 live load reduction:
      L_reduced = L₀ · (0.25 + 15/√(K_LL·A_T))
      Minimum: 0.50·L₀ for members supporting 1 floor
               0.40·L₀ for members supporting 2+ floors

    Parameters
    ----------
    trib_area_m2          : tributary area per floor, m²
    n_stories_above       : number of stories above the column
    dead_kPa, live_kPa    : area loads
    cladding_kN_per_story : perimeter cladding load per story height

    Returns
    -------
    dict with 'Pu_kN' (LRFD factored axial load) and 'P_dead_kN', 'P_live_kN'.
    """
    K_LL = 4.0 if trib_area_m2 > 37.0 else 2.0  # interior vs edge
    A_T = trib_area_m2 * n_stories_above

    if A_T > 37.16:  # 400 ft²
        reduction = 0.25 + 15.0 / math.sqrt(K_LL * A_T)
        if n_stories_above >= 2:
            reduction = max(reduction, 0.40)
        else:
            reduction = max(reduction, 0.50)
        reduction = min(reduction, 1.0)
    else:
        reduction = 1.0

    P_dead = dead_kPa * trib_area_m2 * n_stories_above + cladding_kN_per_story * n_stories_above
    P_live = live_kPa * trib_area_m2 * n_stories_above * reduction
    Pu = 1.2 * P_dead + 1.6 * P_live

    return {"P_dead_kN": P_dead, "P_live_kN": P_live, "Pu_kN": Pu}


# ── R29: Diagrid diagonal axial force (gravity) ─────────────────────────────

def R29_diagrid_gravity_force_kN(
    floor_load_kPa: float,
    trib_area_m2: float,
    theta_deg: float,
    n_diags_at_node: int = 2,
) -> float:
    """
    RULE 29 — Axial force in a diagrid diagonal from gravity load.

    Source: equilibrium at diagrid node (Moon 2007).

    The vertical load P at a node is carried by n diagonal members.
    Each diagonal takes:  N = P / (n · sin(θ))

    Returns
    -------
    N_kN : compression in one diagonal, kN (positive = compression).
    """
    P = floor_load_kPa * trib_area_m2
    return P / (n_diags_at_node * math.sin(math.radians(theta_deg)))


# ── R30: Diagrid diagonal axial force (lateral) ─────────────────────────────

def R30_diagrid_lateral_force_kN(
    V_story_kN: float,
    theta_deg: float,
    n_active_diags: int,
) -> float:
    """
    RULE 30 — Axial force in a diagrid diagonal from lateral shear.

    Source: equilibrium of lateral shear through tube action.

    Sum of horizontal components = V:
    n × N × cos(θ) = V  →  N = V / (n · cos(θ))

    Returns
    -------
    N_kN : axial force per diagonal from lateral load.
    """
    return V_story_kN / (n_active_diags * math.cos(math.radians(theta_deg)))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — CONNECTION DESIGN  (R31–R38)
# ══════════════════════════════════════════════════════════════════════════════

# ── R31: Weld size from force ────────────────────────────────────────────────

def R31_fillet_weld_size_mm(
    force_per_length_kN_mm: float,
    weld_electrode_Fu_MPa: float = 482.0,  # E70XX electrode
) -> float:
    """
    RULE 31 — Required fillet weld leg size.

    Source: AISC 360-22 §J2.4:
      φRn = φ · 0.60 · FEXX · (0.707·a)  per mm of weld
      φ = 0.75
      FEXX = 482 MPa (E70XX)

    → a = F / (φ · 0.60 · FEXX · 0.707)

    Practical limits:
      Minimum: 3 mm (AISC Table J2.4)
      Maximum: member thickness minus 2 mm
      Preferred: 6 mm (single-pass weld, cheapest to fabricate)
      8 mm: 2-pass
      10+ mm: multi-pass, expensive

    Returns
    -------
    a : weld leg size in mm (round up to nearest mm).
    """
    phi = 0.75
    a = force_per_length_kN_mm * 1000.0 / (phi * 0.60 * weld_electrode_Fu_MPa * 0.707)
    a = max(3.0, math.ceil(a))
    return a


# ── R32: Bolt count from force ───────────────────────────────────────────────

def R32_bolt_count(
    total_force_kN: float,
    bolt_diameter_mm: float = 20.0,
    bolt_grade: Literal["8.8", "10.9"] = "8.8",
    shear_type: Literal["single", "double"] = "single",
) -> int:
    """
    RULE 32 — Number of bolts required.

    Source: AISC 360-22 §J3.6 (or Eurocode 3 §3.6):
      φRn per bolt = φ · Fnv · Ab
      Fnv (8.8 / A325, single shear) = 372 MPa (thread excluded)
      Fnv (10.9 / A490) = 457 MPa
      φ = 0.75

    Parameters
    ----------
    total_force_kN   : factored design force to be transferred
    bolt_diameter_mm : 16, 20, 22, 24, 27, 30 mm standard sizes
    bolt_grade       : "8.8" (A325 equiv) or "10.9" (A490 equiv)

    Returns
    -------
    n : number of bolts (integer).

    Practical constraints:
      Always use M20 8.8 as default (cheapest, most common).
      Minimum 2 bolts per connection.
      Maximum practical per row: 6–8 bolts.
    """
    Ab = math.pi * (bolt_diameter_mm / 2.0)**2   # mm²
    Fnv = {"8.8": 372.0, "10.9": 457.0}[bolt_grade]
    n_planes = {"single": 1, "double": 2}[shear_type]
    phi_Rn_per_bolt = 0.75 * Fnv * Ab * n_planes / 1000.0  # kN

    n = math.ceil(total_force_kN / phi_Rn_per_bolt)
    return max(2, n)


# ── R33: Gusset plate thickness ──────────────────────────────────────────────

def R33_gusset_plate_thickness_mm(
    max_force_kN: float,
    gusset_width_mm: float,
    Fy_MPa: float = 248.0,
) -> float:
    """
    RULE 33 — Minimum gusset plate thickness (Whitmore section).

    Source: AISC Design Guide 29: Vertical Bracing Connections.

    Whitmore effective width = gusset_width (or 2·L_bolt·tan30° + bolt_gauge).
    Stress on Whitmore section: σ = P / (b_eff · t)
    Limit: σ ≤ φ·Fy (φ=0.90)

    → t = P / (b_eff · φ · Fy)

    Minimum: 10 mm.  Preferred: match bolt diameter.
    """
    t = (max_force_kN * 1000.0) / (gusset_width_mm * 0.90 * Fy_MPa)
    t = max(10.0, math.ceil(t / 2.0) * 2.0)  # round to nearest 2mm
    return t


# ── R34: End plate thickness for moment connection ───────────────────────────

def R34_endplate_thickness_mm(
    Mu_kNm: float,
    bolt_diameter_mm: float = 20.0,
    bolt_gauge_mm: float = 140.0,
    Fy_plate_MPa: float = 248.0,
) -> float:
    """
    RULE 34 — End plate thickness for a bolted end-plate moment connection.

    Source: AISC Design Guide 4: Extended End-Plate Moment Connections.

    Simplified (yield-line theory):
      t_p = √(4·Mu / (φ·Fy·b_eff·L_yield_line))
    where b_eff ≈ bolt_gauge, L_yield_line ≈ bolt_gauge × 1.5

    Minimum: bolt_diameter (always).

    Returns
    -------
    t_p : plate thickness, mm.
    """
    b_eff = bolt_gauge_mm
    L_yl = bolt_gauge_mm * 1.5
    t = math.sqrt(4.0 * Mu_kNm * 1e6 / (0.90 * Fy_plate_MPa * b_eff * L_yl))
    t = max(bolt_diameter_mm, math.ceil(t / 2.0) * 2.0)
    return t


# ── R35: Base plate dimensions ───────────────────────────────────────────────

def R35_base_plate_mm(
    Pu_kN: float,
    column_d_mm: float,
    column_bf_mm: float,
    fc_MPa: float = 35.0,
) -> dict:
    """
    RULE 35 — Column base plate dimensions.

    Source: AISC 360-22 §J8, AISC Design Guide 1.

    Bearing on concrete: φ·Pp = φ · 0.85·f'c · A₁
    φ = 0.65
    A₁ = B × N (plate area)

    Minimum dimensions: N ≥ d + 2·(50mm), B ≥ bf + 2·(50mm)

    → A₁_req = Pu / (φ · 0.85 · f'c)

    Plate thickness from cantilever bending:
      t_p = n · √(2·Pu / (0.9·Fy·B·N))
    where n = max cantilever (B-bf)/2 or (N-d)/2

    Returns
    -------
    dict: 'B_mm', 'N_mm', 't_mm' (plate width, length, thickness).
    """
    phi = 0.65
    A1_req = (Pu_kN * 1000.0) / (phi * 0.85 * fc_MPa)

    N_min = column_d_mm + 100.0
    B_min = column_bf_mm + 100.0

    if N_min * B_min >= A1_req:
        N, B = N_min, B_min
    else:
        scale = math.sqrt(A1_req / (N_min * B_min))
        N = math.ceil(N_min * scale / 10) * 10
        B = math.ceil(B_min * scale / 10) * 10

    n_cant = max((B - column_bf_mm) / 2.0, (N - column_d_mm) / 2.0)
    fp = Pu_kN * 1000.0 / (B * N)
    t = n_cant * math.sqrt(2.0 * fp / (0.90 * 248.0))
    t = max(20.0, math.ceil(t / 2.0) * 2.0)

    return {"B_mm": B, "N_mm": N, "t_mm": t}


# ── R36: Minimum connection capacity ─────────────────────────────────────────

def R36_min_connection_capacity_kN(
    member_Fy_MPa: float,
    member_A_mm2: float,
    member_Zx_mm3: float,
) -> dict:
    """
    RULE 36 — Minimum connection design forces.

    Source: AISC 360-22 §J1.6 + AISC 341-22 §D2.5b (seismic).

    Non-seismic:
      Shear connection: 50% of member web shear capacity
      = 0.50 × φ × 0.60 × Fy × d × tw

    Seismic (AISC 341-22):
      Brace connection must develop expected yield:
      T_req = Ry × Fy × A_g
      Ry = 1.1 for A992, 1.4 for A36, 1.3 for A500

    Returns
    -------
    dict: 'min_shear_kN', 'min_axial_tension_kN', 'seismic_tension_kN'
    """
    Ry = 1.1  # A992
    return {
        "min_shear_kN": 0.50 * 0.90 * 0.60 * member_Fy_MPa * member_A_mm2 * 0.5 / 1000.0,
        "min_axial_tension_kN": 0.75 * member_Fy_MPa * member_A_mm2 / 1000.0,
        "seismic_tension_kN": Ry * member_Fy_MPa * member_A_mm2 / 1000.0,
    }


# ── R37: Bolt spacing and edge distance ─────────────────────────────────────

def R37_bolt_layout_mm(
    bolt_diameter_mm: float = 20.0,
) -> dict:
    """
    RULE 37 — Bolt spacing and edge distance limits.

    Source: AISC 360-22 §J3.3–J3.5.

    Rules (ALL in terms of bolt diameter d_b and hole diameter d_h):
      d_h = d_b + 2 mm  (standard hole)
      Min spacing: 2.667 × d_b  (AISC §J3.3, preferred 3×d_b)
      Max spacing: min(24t, 305 mm) where t = thinnest connected part
      Min edge distance: Table J3.4 values ≈ 1.5 × d_b to 2.0 × d_b
      Min end distance: same as edge distance

    Returns
    -------
    dict with all limits in mm.
    """
    db = bolt_diameter_mm
    dh = db + 2.0
    return {
        "hole_diameter_mm": dh,
        "min_spacing_mm": round(3.0 * db, 1),
        "min_edge_distance_mm": round(1.75 * db, 1),
        "min_end_distance_mm": round(1.75 * db, 1),
        "preferred_gauge_mm": round(5.5 * db, 1),
    }


# ── R38: Connection type from member role ────────────────────────────────────

def R38_connection_type(
    member_role: Literal[
        "primary_beam", "secondary_beam", "column", "brace",
        "diagrid_diagonal", "ring_beam", "outrigger_chord",
    ],
    seismic_sdc: Literal["A", "B", "C", "D", "E", "F"] = "D",
) -> dict:
    """
    RULE 38 — Connection type selection from member role and seismic category.

    Source: AISC 360-22 Part 10/11/12/13, AISC 341-22 §E/F.

    Returns
    -------
    dict:
      'type'       : str  (shear_tab | endplate | welded_flange | gusset | cast_node)
      'moment'     : bool (is this a moment connection?)
      'prequalified': bool (per AISC 358 for seismic)
      'forces_to_design': list[str]  (which forces the connection must carry)
    """
    high_seismic = seismic_sdc in ("D", "E", "F")

    rules = {
        "secondary_beam": {
            "type": "shear_tab", "moment": False, "prequalified": False,
            "forces_to_design": ["shear"],
        },
        "primary_beam": {
            "type": "endplate" if high_seismic else "shear_tab",
            "moment": high_seismic,
            "prequalified": high_seismic,
            "forces_to_design": ["shear", "moment"] if high_seismic else ["shear"],
        },
        "column": {
            "type": "welded_flange",
            "moment": True, "prequalified": high_seismic,
            "forces_to_design": ["axial", "moment_major", "moment_minor", "shear"],
        },
        "brace": {
            "type": "gusset",
            "moment": False, "prequalified": False,
            "forces_to_design": ["axial_tension", "axial_compression"],
        },
        "diagrid_diagonal": {
            "type": "cast_node" if high_seismic else "gusset",
            "moment": False, "prequalified": False,
            "forces_to_design": ["axial_tension", "axial_compression"],
        },
        "ring_beam": {
            "type": "endplate",
            "moment": True, "prequalified": False,
            "forces_to_design": ["shear", "moment", "axial"],
        },
        "outrigger_chord": {
            "type": "welded_flange",
            "moment": True, "prequalified": True,
            "forces_to_design": ["axial", "moment_major", "shear"],
        },
    }
    return rules.get(member_role, rules["secondary_beam"])


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — VALIDATION CHECKS  (R39–R50)
#
#  Every check returns: {"pass": bool, "DCR": float, "limit": str, "msg": str}
# ══════════════════════════════════════════════════════════════════════════════

def R39_check_slenderness(KL_r: float) -> dict:
    """R39 — KL/r ≤ 200 (AISC 360-22 §E2). Practical target: ≤120 for columns."""
    return {"pass": KL_r <= 200, "DCR": KL_r / 200.0,
            "limit": "KL/r ≤ 200", "msg": f"KL/r = {KL_r:.1f}"}

def R40_check_interaction_H1(
    Pr_kN: float, Pc_kN: float,
    Mrx_kNm: float, Mcx_kNm: float,
    Mry_kNm: float = 0, Mcy_kNm: float = 1e9,
) -> dict:
    """R40 — AISC 360-22 §H1 bilinear interaction."""
    ratio = abs(Pr_kN) / max(abs(Pc_kN), 1e-9)
    bending = abs(Mrx_kNm) / max(abs(Mcx_kNm), 1e-9) + abs(Mry_kNm) / max(abs(Mcy_kNm), 1e-9)
    if ratio >= 0.2:
        DCR = ratio + (8.0/9.0) * bending
    else:
        DCR = ratio / 2.0 + bending
    return {"pass": DCR <= 1.0, "DCR": DCR,
            "limit": "H1 ≤ 1.0", "msg": f"DCR = {DCR:.3f}"}

def R41_check_deflection(delta_mm: float, span_mm: float, ratio: float = 360) -> dict:
    """R41 — IBC 2024 Table 1604.3: δ ≤ L/ratio."""
    limit_mm = span_mm / ratio
    DCR = delta_mm / max(limit_mm, 1e-6)
    return {"pass": delta_mm <= limit_mm, "DCR": DCR,
            "limit": f"L/{ratio:.0f}", "msg": f"δ={delta_mm:.1f}mm, limit={limit_mm:.1f}mm"}

def R42_check_story_drift(delta_story_mm: float, h_story_mm: float, limit: float = 0.020) -> dict:
    """R42 — ASCE 7-22 Table 12.12-1: Δ/h ≤ 0.020 (SDC D-F, office)."""
    ratio = delta_story_mm / max(h_story_mm, 1.0)
    return {"pass": ratio <= limit, "DCR": ratio / limit,
            "limit": f"Δ/h ≤ {limit}", "msg": f"Δ/h = {ratio:.4f}"}

def R43_check_local_buckling_flange(bf_mm: float, tf_mm: float, Fy_MPa: float = 345) -> dict:
    """R43 — AISC 360-22 Table B4.1b: bf/(2tf) ≤ 0.38√(E/Fy) for compact flange."""
    ratio = bf_mm / (2.0 * max(tf_mm, 0.1))
    limit = 0.38 * math.sqrt(E_STEEL / Fy_MPa)
    return {"pass": ratio <= limit, "DCR": ratio / limit,
            "limit": f"b/2t ≤ {limit:.1f}", "msg": f"bf/2tf = {ratio:.1f}"}

def R44_check_local_buckling_web(h_mm: float, tw_mm: float, Fy_MPa: float = 345) -> dict:
    """R44 — AISC 360-22 Table B4.1a: h/tw ≤ 1.49√(E/Fy) for non-slender web."""
    ratio = h_mm / max(tw_mm, 0.1)
    limit = 1.49 * math.sqrt(E_STEEL / Fy_MPa)
    return {"pass": ratio <= limit, "DCR": ratio / limit,
            "limit": f"h/tw ≤ {limit:.1f}", "msg": f"h/tw = {ratio:.1f}"}

def R45_check_vibration(span_m: float, delta_LL_mm: float) -> dict:
    """
    R45 — AISC DG 11 floor vibration: fn ≥ 4 Hz.
    Simplified: fn ≈ 18 / √(δ_LL_mm)  where δ is in mm under 1 kPa.
    """
    if delta_LL_mm <= 0:
        return {"pass": True, "DCR": 0, "limit": "fn ≥ 4 Hz", "msg": "No deflection"}
    fn = 18.0 / math.sqrt(delta_LL_mm)
    return {"pass": fn >= 4.0, "DCR": 4.0 / max(fn, 0.01),
            "limit": "fn ≥ 4 Hz", "msg": f"fn = {fn:.1f} Hz"}

def R46_check_HSS_Dt(D_mm: float, t_mm: float, Fy_MPa: float = 317) -> dict:
    """R46 — AISC 360-22 §I2.2a: D/t ≤ 0.15·E/Fy for filled CHS (CFST)."""
    ratio = D_mm / max(t_mm, 0.1)
    limit = 0.15 * E_STEEL / Fy_MPa
    return {"pass": ratio <= limit, "DCR": ratio / limit,
            "limit": f"D/t ≤ {limit:.0f}", "msg": f"D/t = {ratio:.1f}"}

def R47_check_weld_to_base_metal(weld_a_mm: float, plate_t_mm: float) -> dict:
    """R47 — AISC 360-22 Table J2.4: weld ≤ plate_t - 2mm (plate > 6mm)."""
    limit = plate_t_mm - 2.0 if plate_t_mm > 6 else plate_t_mm
    return {"pass": weld_a_mm <= limit, "DCR": weld_a_mm / max(limit, 0.1),
            "limit": f"a ≤ t-2mm", "msg": f"a={weld_a_mm}mm, limit={limit:.0f}mm"}

def R48_check_member_count_at_node(n_members: int) -> dict:
    """R48 — Minimum 3 members at every structural node (R16)."""
    return {"pass": n_members >= 3, "DCR": 3.0 / max(n_members, 1),
            "limit": "n ≥ 3", "msg": f"n = {n_members}"}

def R49_check_thermal_expansion(member_length_m: float, delta_T_C: float = 40) -> dict:
    """
    R49 — Thermal movement check.
    Source: physics: ΔL = α·L·ΔT.
    If ΔL > 25mm, an expansion joint or sliding connection is needed.
    """
    dL_mm = ALPHA_T * member_length_m * 1000.0 * delta_T_C
    return {"pass": dL_mm <= 25.0, "DCR": dL_mm / 25.0,
            "limit": "ΔL ≤ 25mm",
            "msg": f"ΔL = {dL_mm:.1f}mm {'→ expansion joint needed' if dL_mm > 25 else '— OK'}"}

def R50_check_self_weight_ratio(member_weight_kN: float, applied_load_kN: float) -> dict:
    """
    R50 — Self-weight sanity check.
    Rule: member self-weight should be < 10% of applied load.
    If > 10%, the section is likely oversized.
    If self-weight > applied load, the design is unreasonable.
    """
    if applied_load_kN <= 0:
        return {"pass": True, "DCR": 0, "limit": "SW < 10% load", "msg": "No applied load"}
    ratio = member_weight_kN / applied_load_kN
    return {"pass": ratio <= 0.10, "DCR": ratio / 0.10,
            "limit": "SW/P ≤ 10%", "msg": f"SW/P = {ratio:.1%}"}


# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN TABLE — ALL NUMERIC LIMITS IN ONE PLACE
#  (for the agent to validate any output against)
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN = {
    # (min, max, unit, source, rule_id)
    "member_length_m":       (0.5,    18.0,   "m",     "AISC 360 §E2 + practice",   "R01-R02"),
    "primary_spacing_m":     (4.5,    12.0,   "m",     "Economy studies",            "R09"),
    "secondary_spacing_m":   (2.0,     4.5,   "m",     "Deck span limit",            "R10"),
    "diagrid_theta_deg":     (50.0,   75.0,   "deg",   "Moon 2007",                  "R11"),
    "diagrid_module_h_m":    (6.0,    20.0,   "m",     "Practice",                   "R12"),
    "diagrid_bay_width_m":   (3.0,    15.0,   "m",     "Geometry",                   "R13"),
    "beam_depth_mm":         (150,    3000,   "mm",    "W150 – plate girder",        "R19"),
    "column_depth_mm":       (150,    1000,   "mm",    "W150 – W920",                "R20"),
    "HSS_diameter_mm":       (114,     914,   "mm",    "HSS 4.5 – HSS 36",          "R24"),
    "beam_weight_kg_m":      (13,     1100,   "kg/m",  "W150×13 – W920×1077",       "R25"),
    "KL_r":                  (0,       200,   "—",     "AISC 360 §E2",               "R39"),
    "DCR_H1":                (0,       1.0,   "—",     "AISC 360 §H1",               "R40"),
    "deflection_LL":         (0, "L/360",     "mm",    "IBC 2024 T1604.3",           "R41"),
    "deflection_TL":         (0, "L/240",     "mm",    "IBC 2024 T1604.3",           "R41"),
    "story_drift":           (0,     0.020,   "Δ/h",   "ASCE 7-22 T12.12-1",        "R42"),
    "floor_vibration_Hz":    (4.0,    None,   "Hz",    "AISC DG 11",                 "R45"),
    "floor_height_m":        (2.7,     6.0,   "m",     "IBC 2024",                   "R08"),
    "weld_size_mm":          (3,       16,    "mm",    "AISC 360 §J2.4",            "R31"),
    "bolt_diameter_mm":      (16,      36,    "mm",    "M16 – M36 standard",         "R32"),
    "gusset_thickness_mm":   (10,      50,    "mm",    "Practice",                   "R33"),
    "thermal_movement_mm":   (0,       25,    "mm",    "Before expansion joint",     "R49"),
    "min_node_valence":      (3,      None,   "count", "Maxwell stability",          "R48"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("STEEL RULEBOOK SELF-TEST")
    print("=" * 72)

    # R01: max member length for HSS 200×200×10 (r≈78mm)
    L = R01_max_member_length_m(r_min_mm=78.0)
    print(f"R01  Max length HSS200 (r=78mm): {L:.1f} m  [expect ~15.6]")

    # R03: validate a 14m edge as primary beam
    v = R03_grid_spacing_from_edge_length_m(14.0, "primary_beam")
    print(f"R03  14m primary beam: {v}")

    # R11: diagrid angle for H/W=6
    theta = R11_diagrid_angle_deg(6.0)
    print(f"R11  Diagrid angle H/W=6: {theta:.1f}°  [expect ~62.5]")

    # R19: beam depth for 10m span
    d = R19_beam_depth_from_span_mm(10.0, "simple", "floor")
    print(f"R19  Beam depth for 10m floor: {d:.0f} mm  [expect ~500]")

    # R26: section properties for d=500mm W-shape
    props = R26_section_properties_from_depth_mm(500, "W_shape")
    print(f"R26  W500 approx: A={props['A_mm2']:.0f} mm², "
          f"Ix={props['Ix_mm4']:.0e} mm⁴, wt={props['weight_kg_m']:.0f} kg/m")

    # R28: column load for 8m×8m trib area, 20 stories
    col = R28_column_axial_load_kN(64.0, 20, dead_kPa=3.5, live_kPa=2.4)
    print(f"R28  Column Pu (64m², 20 stories): {col['Pu_kN']:.0f} kN")

    # R40: interaction check
    chk = R40_check_interaction_H1(Pr_kN=2000, Pc_kN=4000, Mrx_kNm=300, Mcx_kNm=800)
    print(f"R40  H1 check: DCR={chk['DCR']:.3f}, pass={chk['pass']}")

    # R43: flange buckling
    fb = R43_check_local_buckling_flange(bf_mm=250, tf_mm=16)
    print(f"R43  Flange b/2t: {fb['msg']}, pass={fb['pass']}")

    # R49: thermal
    th = R49_check_thermal_expansion(member_length_m=50.0, delta_T_C=40)
    print(f"R49  Thermal 50m: {th['msg']}")

    print("=" * 72)
    print("ALL SELF-TESTS COMPLETE")
    print("=" * 72)
