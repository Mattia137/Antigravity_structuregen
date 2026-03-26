# Complex Non-Hierarchical Structural Systems — AI Design Manual
## Integrated Reference: IBC 2024 / ASCE 7-22 / AISC 360-22 / ACI 318-19

---

## Chapter 1 — Geometry Interpretation Protocol

### 1.1 Mesh-to-Structure Mapping
When reading a polyhedral mesh:
- **Every mesh vertex** → primary structural node. Coordinates are EXACT; do not round or relocate.
- **Every mesh edge with dihedral angle > 5°** → `primary_crease` member (column, beam, arch rib, or ridge).
- **Every mesh edge with dihedral angle ≤ 5°** → `secondary_lattice` member (bracing, tie, sub-purlin).
- **Mesh faces** → implied floor/roof panels; no structural members need to be added inside flat faces unless buckling checks require it.

### 1.2 Primary Crease Classification
After angle classification, further refine by geometry:
| Condition | Role | Default Section |
|---|---|---|
| Nearly vertical (Z angle > 70°) | Column | W14x90 or W14x283 |
| Sloped 20°–70°, long span > 8 m | Raking column / arch rib | W12x50 or W14x90 |
| Nearly horizontal (Z angle < 20°) | Beam or floor beam | IPE_300 or W12x50 |
| Short horizontal < 3 m | Tie beam | HEA_200 |
| Diagonal in perimeter plane | Façade brace | Tubular_HSS_4x4x1/4 or HSS6x0.500 |

### 1.3 Secondary Structure Addition Rules
Only add secondary nodes and edges when the primary topology has:
- Unbraced column length L_u > L/360 × 360 = L (i.e., always check)
- Missing lateral bracing at mid-height of columns taller than 6 m
- X-bracing absent between floor levels in any bay > 4 m wide
- Out-of-plane triangulation missing on curved surface panels > 4 m²
- Any primary member spanning > 12 m without mid-point lateral support

---

## Chapter 2 — Load Combinations (ASCE 7-22 / IBC 2024)

### 2.1 LRFD Strength Load Combinations
```
1.  1.4D
2.  1.2D + 1.6L + 0.5(Lr or S or R)
3.  1.2D + 1.6(Lr or S or R) + (L or 0.5W)
4.  1.2D + 1.0W + L + 0.5(Lr or S or R)
5.  0.9D + 1.0W
6.  1.2D + 1.0E + L + 0.2S        ← GOVERNING for seismic (SDC D-F)
7.  0.9D + 1.0E
```
**Primary design combo:** `1.2D + 1.0E + 1.0L` (conservative simplification for initial iteration).

### 2.2 Serviceability Load Combinations (ASD)
- **Deflection check:** D + L (unfactored)
- **Drift check:** 0.7E (ASCE 7-22 §12.8.6)

### 2.3 Nominal Load Values (default for unknown occupancy)
| Load | Value |
|---|---|
| Dead load D | Self-weight (auto from section × density) + 1.0 kPa superimposed |
| Live load L | 2.4 kPa (office occupancy, ASCE 7-22 Table 4.3-1) |
| Seismic E | ELF per ASCE 7-22 §12.8 with LA Basin parameters |
| Wind W | Not modeled in initial iteration |

---

## Chapter 3 — Seismic Design: Los Angeles Basin (SDC D–F)

### 3.1 Site Parameters (Default: Site Class D, Coastal LA)
- S_s = 2.50 g (0.2-sec MCE spectral acceleration)
- S_1 = 1.00 g (1.0-sec MCE spectral acceleration)
- F_a = 1.0 (Site Class D, S_s ≥ 1.25)
- F_v = 1.7 (Site Class D, S_1 ≥ 0.60)
- S_DS = (2/3) × F_a × S_s = **1.67 g**
- S_D1 = (2/3) × F_v × S_1 = **1.13 g**

### 3.2 Equivalent Lateral Force (ELF)
```
V = C_s × W
C_s = S_DS / (R / I_e)
```
- R = 8 for Special Moment Frame (SMF) or Special Concentrically Braced Frame (SCBF)
- R = 6 for Ordinary Concentrically Braced Frame (OCBF)
- I_e = 1.0 (standard occupancy), 1.25 (essential facility)

Vertical distribution (ASCE 7-22 §12.8.3):
```
F_x = C_vx × V
C_vx = (w_x × h_x^k) / Σ(w_i × h_i^k)
k = 1.0 if T ≤ 0.5 s; k = 2.0 if T ≥ 2.5 s; interpolate
```

### 3.3 Allowable Story Drift (ASCE 7-22 Table 12.12-1)
| Occupancy | Limit |
|---|---|
| Standard (Risk Cat. II) | 0.020 h_sx |
| Essential (Risk Cat. III/IV) | 0.015 h_sx |

### 3.4 Concrete Shear Core Placement Rules
- **Location:** Center of building or symmetrically placed to minimize eccentricity between center of mass and center of rigidity.
- **Shape:** C-shape or closed tube around elevator/stair core.
- **Wall thickness:** ≥ h_w / 16 but minimum 200 mm (8 in) for SDC D+.
- **Aspect ratio:** Height-to-length ratio 4:1 to 8:1 optimal. Above 10:1 requires coupled-wall design.
- **Minimum core area:** ~1.5% of gross floor area per LABC recommendation for SDC D.

---

## Chapter 4 — Structural System Selection

### 4.1 Decision Tree by Height
```
H ≤ 15 m  →  Simple moment frame + X-bracing
15–50 m   →  Special Moment Frame (SMF) + concrete core
50–150 m  →  Diagrid + outriggers or tube-in-tube
> 150 m   →  Mega-frame + belt truss every 20 floors
```

### 4.2 Decision Tree by Plan Aspect Ratio (L/W)
```
L/W ≤ 1.5  →  Symmetric core, perimeter moment frames
L/W 1.5–3  →  Two cores at ¼ points, lateral ties
L/W > 3    →  Distributed shear walls, minimize torsion
```

### 4.3 Non-Hierarchical (Diagrid / Freeform) Systems
For freeform geometry where the primary creases define an irregular lattice:
- Primary crease members form the main structural topology (no traditional column grid).
- Each crease node is a potential transfer point — assign heavier sections at high-connectivity nodes (degree ≥ 4).
- Secondary lattice triangulates the mesh faces for out-of-plane stability.
- Shear cores must be sited at geometry "valleys" (local minima in plan) for maximum torsional effect.

---

## Chapter 5 — Cross-Section Selection Rules

### 5.1 Steel Wide-Flange (W-shapes)
| Member Role | Span | Recommended Section |
|---|---|---|
| Heavy column (high axial) | Any | W14x283, W14x159 |
| Medium column | Any | W14x90 |
| Long-span beam | > 9 m | W24x146, W18x97 |
| Medium-span beam | 5–9 m | W12x53, W12x50 |
| Short-span beam / tie | < 5 m | IPE_300, W8x31 |

### 5.2 HSS (Hollow Structural Sections)
| Role | Section |
|---|---|
| Diagonal brace, large | HSS16x0.625 |
| Diagonal brace, medium | HSS10x0.500 |
| Diagonal brace, light | HSS6x0.500 |
| Square column | HSS12x12x0.625 |
| Square brace | HSS8x8x0.500, Tubular_HSS_4x4x1/4 |

### 5.3 European Sections
- **IPE_300**: Light beam, span 4–7 m, low gravity load
- **HEA_200**: Brace or secondary beam, span < 5 m

### 5.4 Section Sizing by Span (Quick Rules)
- Beam depth ≈ span / 20 (steel), span / 12 (concrete)
- Column: P_u / (φ × F_y × A) < 0.85 (compression check)
- Brace: KL/r < 200 (AISC 360-22 §E2)

---

## Chapter 6 — Connection Type Rules

### 6.1 Fixed (Moment-Resisting) Connections
Use `"connection": "fixed"` for:
- All beam-to-column connections in moment frames
- Continuous column splices
- Base plate connections at foundation nodes
- Arch rib joints where bending moment transfers

### 6.2 Pinned Connections
Use `"connection": "pinned"` for:
- Brace end connections (gusset plate)
- Secondary lattice-to-primary member connections
- Truss chord connections (where rotation is released)
- Secondary floor beam simple connections

**Rule of Thumb:** Primary creases = fixed. Secondary lattice = pinned. Never leave an isolated node with only pinned connections (creates mechanism).

---

## Chapter 19 — System Topology Assessment

### 19.1 Connectivity Check
For each node, count the degree (number of members framing in):
- Degree ≥ 3: stable in 2D, check 3D
- Degree ≥ 4: fully stable in 3D, good load distribution
- Degree ≤ 2: potential mechanism — add secondary brace or merge node

### 19.2 Redundancy Factor ρ (ASCE 7-22 §12.3.4)
- ρ = 1.0 if removing any single member does not reduce story shear resistance by > 33% AND structure has ≥ 2 bays of lateral resistance in each direction.
- ρ = 1.3 otherwise (increases seismic demands by 30%).

### 19.3 Irregularity Detection
| Irregularity | Trigger | Action |
|---|---|---|
| Torsional | Center of rigidity offset > 20% plan dim. | Redistribute cores or add perimeter braces |
| Soft story | Story stiffness < 70% of story above | Add bracing in weak story |
| Mass irregularity | Floor mass > 150% of adjacent | Rebalance or add outrigger |
| Geometric | Floor area reduces > 50% | Stiffen setback level |

---

## Chapter 21 — Code Compliance Checks

### 21.1 Deflection Limits (Serviceability)
| Member | Limit |
|---|---|
| Floor beam | L/360 (live load only) |
| Roof beam | L/240 (total load) |
| Cantilever | L/180 |
| Column lateral (wind) | H/500 |
| Story drift (seismic) | 0.020 h_sx |

### 21.2 Strength Checks (AISC 360-22 LRFD)
**Tension:** φ_t × F_y × A_g ≥ T_u  (φ_t = 0.90)
**Compression:** φ_c × P_n ≥ P_u    (φ_c = 0.90, per §E3)
**Bending:** φ_b × M_n ≥ M_u        (φ_b = 0.90)
**Shear:** φ_v × V_n ≥ V_u          (φ_v = 1.00)
**Combined:** (P_u/φ_c P_n) + (8/9)(M_ux/φ_b M_nx + M_uy/φ_b M_ny) ≤ 1.0

### 21.3 Stability Checks
- **Column slenderness:** KL/r ≤ 200
- **Beam lateral-torsional buckling:** L_b ≤ L_p (compact unbraced length)
- **Local buckling:** λ ≤ λ_p for compact sections per AISC Table B4.1

### 21.4 Displacement Check Formula
Max displacement must satisfy:
```
δ_max / H_total ≤ 1/500 (wind serviceability)
δ_x = C_d × δ_xe / I_e ≤ Δ_a (seismic, ASCE 7-22 §12.8.6)
```
where C_d = deflection amplification factor (5.5 for SMF, 5.0 for SCBF).

---

## Appendix A — Sustainability Data (2025)

```json
{
  "structural_steel": {
    "embodied_carbon_kgCO2e_per_kg": 1.22,
    "cost_usd_per_ton": 2653.00,
    "notes": "Hot-rolled steel, 2025 EPD average, 11% reduction from 2021"
  },
  "reinforced_concrete": {
    "embodied_carbon_kgCO2e_per_kg": 0.20,
    "cost_usd_per_yard3": 145.00,
    "notes": "SCM-blend concrete, ~30% cement replacement"
  },
  "rebar_steel": {
    "embodied_carbon_kgCO2e_per_kg": 0.68,
    "cost_usd_per_ton": 785.00,
    "notes": "90% recycled EAF rebar"
  }
}
```

### Section Mass Reference (kg/m)
| Section | kg/m |
|---|---|
| W8x31 | 46.2 |
| W12x50 | 74.4 |
| W12x53 | 78.9 |
| W14x90 | 133.9 |
| W14x159 | 236.7 |
| W14x283 | 421.4 |
| W18x97 | 144.3 |
| W24x146 | 217.3 |
| IPE_300 | 42.2 |
| HEA_200 | 42.3 |
| HSS6x0.500 | 37.0 |
| HSS8x8x0.500 | 76.9 |
| HSS10x0.500 | 48.0 |
| HSS12x12x0.625 | 121.0 |
| HSS16x0.625 | 61.8 |
| Tubular_HSS_4x4x1/4 | 25.0 |

---

## Appendix B — Cross-Section Properties Library

All values in imperial units: A (in²), Iy (in⁴), Iz (in⁴), J (in⁴)

### Wide-Flange Steel (W-shapes, AISC)
| Section | A | Iy | Iz | J |
|---|---|---|---|---|
| W8x31 | 9.13 | 37.1 | 110.0 | 0.54 |
| W12x50 | 14.6 | 56.3 | 391.0 | 1.04 |
| W12x53 | 15.6 | 95.8 | 425.0 | 1.58 |
| W14x90 | 26.5 | 362.0 | 999.0 | 4.06 |
| W14x159 | 46.7 | 748.0 | 1900.0 | 14.5 |
| W14x283 | 83.3 | 1070.0 | 3840.0 | 72.8 |
| W18x97 | 28.5 | 201.0 | 1750.0 | 5.86 |
| W24x146 | 43.0 | 391.0 | 4580.0 | 16.2 |
| IPE_300 | 8.34 | 14.5 | 201.0 | 0.48 |
| HEA_200 | 8.35 | 32.2 | 88.7 | 0.35 |

### HSS Round (pipe-like, AISC HSS)
| Section | A | Iy | Iz | J |
|---|---|---|---|---|
| HSS6x0.500 | 8.09 | 24.7 | 24.7 | 49.4 |
| HSS10x0.500 | 14.0 | 120.0 | 120.0 | 240.0 |
| HSS16x0.625 | 29.1 | 670.0 | 670.0 | 1340.0 |

### HSS Square (AISC)
| Section | A | Iy | Iz | J |
|---|---|---|---|---|
| Tubular_HSS_4x4x1/4 | 3.37 | 7.8 | 7.8 | 12.5 |
| HSS8x8x0.500 | 13.5 | 105.0 | 105.0 | 169.0 |
| HSS12x12x0.625 | 26.9 | 420.0 | 420.0 | 676.0 |

---

## Appendix D — AI Output JSON Schema

The AI must return ONLY the following JSON structure (no markdown, no extra text):

```json
{
  "nodes": [{"id": 0, "x": 0.0, "y": 0.0, "z": 0.0}],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "type": "primary_crease|secondary_lattice",
      "section": "W14x90",
      "connection": "fixed|pinned"
    }
  ],
  "cores": [{"x": 0.0, "y": 0.0, "thickness": 0.3}]
}
```

### Schema Rules:
1. ALL primary_nodes must appear in `nodes` with EXACT coordinates (copy verbatim).
2. ALL primary_edges must appear in `edges` with their assigned `section` and `connection`.
3. Secondary nodes start at ID = max(primary_node_ids) + 1.
4. Every edge MUST have a `section` from the approved list and a `connection` of `"fixed"` or `"pinned"`.
5. `cores` list may be empty `[]` if no shear core is warranted.
6. Do NOT include any text outside the JSON object.
