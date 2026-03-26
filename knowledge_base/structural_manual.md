# Structural Engineering Manual: Generative Design Architecture

## 1. Complex Lattices and Load-Path Logic

### Non-Hierarchical Steel Lattices (Diagrid and Shell Structures)
Non-hierarchical steel lattices, commonly known as diagrid (diagonal grid) structures, efficiently manage load paths predominantly through the axial action (tension and compression) of their diagonal members rather than bending moments. This eliminates the need for conventional vertical columns, allowing the structural skin to distribute forces across a unified network.

- **Primary Creases (Major Loads):** In non-hierarchical shells (e.g., BMW Welt, Gehry's Barcelona Fish), the primary creases act as the main load-bearing arches or ridges. These major structural lines attract and transfer the dominant gravity and lateral loads directly down to the foundation via axial compression.
- **Secondary Triangulations (Buckling Stability):** The secondary interlocking triangles connecting the primary creases provide redundancy, out-of-plane stiffness, and bracing. They prevent localized buckling of the primary members and ensure that lateral shear forces (wind/seismic) are effectively dissipated across the facade via tension-compression couples.

## 2. LABC & ASCE 7-22 Seismic Design (Los Angeles Basin)

For structures in the Los Angeles Basin assigned to Seismic Design Category (SDC) D and E, ASCE 7-22 and the Los Angeles Building Code (LABC) dictate strict Equivalent Lateral Force (ELF) parameters:

### Seismic Base Shear Formula
The total design lateral force (base shear), $V$, is calculated as:
$$V = C_s W$$
Where $W$ is the effective seismic weight of the structure, and $C_s$ is the seismic response coefficient:
$$C_s = \frac{S_{DS}}{(R/I_e)}$$

### Site-Adjusted Parameters for LA
- **$S_{DS}$ (Design Spectral Acceleration at Short Periods):** $S_{DS} = \frac{2}{3} S_{MS}$ where $S_{MS} = F_a S_s$
- **$S_{D1}$ (Design Spectral Acceleration at 1-Second Period):** $S_{D1} = \frac{2}{3} S_{M1}$ where $S_{M1} = F_v S_1$
- **R (Response Modification Coefficient):** Defines ductility. Higher values require stringent joint detailing.
- **$I_e$ (Importance Factor):** $1.0$ for standard, $1.25$ or $1.5$ for essential facilities.

### Concrete Shear Core Placement Rules
Shear core design relies heavily on building size (square footage) and height (story count) to act as the primary Resisting System against $V$:
- **Placement Logic:** For high-rises, shear cores are centralized in a "C-shape" or boxed configuration surrounding elevator/stair shafts to maximize torsional rigidity.
- **Rule of Thumb by Story Count:** As height increases, the core wall thickness typically must satisfy $h/16$, with absolute minimums of 150 mm (6 inches) per ACI 318.
- **Rule of Thumb by Square Footage:** In large floorplate structures, if an exterior box system is insufficient to limit inter-story drift to allowable LABC limits (usually $0.020 h_{sx}$ to $0.025 h_{sx}$), secondary distributed interior shear walls must supplement the core. 
- **Aspect Ratio Limit:** The height-to-length/width ratio of the core dictates whether bending or shear dominates; optimized core ratios range between 4 and 8.

## 3. Sustainability and Cost Metadata

The following JSON dictionary outlines the estimated 2025 Embodied Carbon Coefficients ($kgCO_2e/kg$) and Material Costs for structural steel and reinforced concrete:

```json
{
    "structural_steel": {
        "embodied_carbon_kgCO2e_per_kg": 1.22,
        "cost_usd_per_ton": 2653.00,
        "notes": "Based on 2025 hot-rolled steel projections averaging an 11% reduction from 2021 A1-A3 EPDs. Costs reflect localized tariffs."
    },
    "reinforced_concrete": {
        "embodied_carbon_kgCO2e_per_kg": 0.20,
        "cost_usd_per_yard": 145.00,
        "notes": "Composite value based on standard strength concrete mixes replacing cement with SCMs (e.g. slag). Cost is per cubic yard."
    },
    "rebar_steel": {
        "embodied_carbon_kgCO2e_per_kg": 0.68,
        "cost_usd_per_ton": 785.00,
        "notes": "Based on 90% recycled fraction rebar reinforcement."
    }
}
```
