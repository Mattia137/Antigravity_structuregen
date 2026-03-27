"""
FEM SOLVER & ADVANCED GEOMETRY — EXTENSIONS FOR steel_mesh_structural_rules.py
================================================================================
Implements the three major upgrades:

  Module A — Direct-Stiffness FEM Solver
      Full 3D frame assembler (12-DOF Euler–Bernoulli beam elements) that
      produces actual Pr, Mr, δ values to feed into the existing code-check
      functions from Section 11.

  Module B — Cotangent-Weight Laplace–Beltrami Curvature
      Replaces the placeholder curvature estimators in Section 1 with the
      mathematically rigorous cotangent-weight discrete Laplace–Beltrami
      operator for both mean and Gaussian curvature.

  Module C — Mesh Surface Projection for Diagrid Layout
      Adds the projection step in generate_diagrid_layout to map the parametric
      grid onto arbitrary curved mesh surfaces using closest-point projection
      with barycentric interpolation.

Dependencies: numpy, scipy (sparse solvers)
Codes: AISC 360-22 · ASCE 7-22 · ACI 318-19

Author: Structural Engineering AI Pipeline
"""

from __future__ import annotations
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import Optional, Literal

# ─── Import the parent module's constants and types ──────────────────────────
from steel_mesh_structural_rules import (
    E_STEEL, G_STEEL, Fy_A992, Fy_A500C,
    SLENDERNESS_MAX, DRIFT_SEISMIC_SDC_DF, DEFLECTION_LL_LIMIT,
    DEFLECTION_TL_LIMIT, DEFLECTION_ROOF_LL,
    MeshDescriptors, DiagridNode, DiagridMember, DiagridLayout,
    CodeCheckResult,
    aisc_360_interaction_H1, column_fcr_aisc360_E3,
    check_member_interaction, check_slenderness, check_story_drift,
    check_beam_deflection,
    diagrid_theta_from_HW, diagrid_module_height_m,
    diagrid_bay_count, diagrid_diagonal_length_m,
    _perimeter_xy,
)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE A — DIRECT-STIFFNESS FEM SOLVER  (3D Frame Elements)
# ══════════════════════════════════════════════════════════════════════════════
#
# Theory: 12-DOF Euler–Bernoulli beam element in 3D space.
# Each node has 6 DOF: [ux, uy, uz, θx, θy, θz]
# The element stiffness matrix is assembled in global coordinates using a
# rotation matrix derived from the element's local axis and a reference
# "web direction" vector.
#
# References:
#   - McGuire, Gallagher & Ziemian, "Matrix Structural Analysis" (2nd ed.)
#   - Kassimali, "Matrix Analysis of Structures" (3rd ed.)
#   - AISC 360-22 §C1 — General stability requirements
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameSection:
    """Cross-section properties for a frame element."""
    name:   str
    A_in2:  float      # gross area, in²
    Ix_in4: float      # major-axis moment of inertia, in⁴
    Iy_in4: float      # minor-axis moment of inertia, in⁴
    J_in4:  float      # torsional constant, in⁴
    Zx_in3: float      # plastic section modulus (major), in³
    Zy_in3: float      # plastic section modulus (minor), in³
    rx_in:  float      # radius of gyration (major), in
    ry_in:  float      # radius of gyration (minor), in
    d_in:   float      # overall depth, in
    bf_in:  float      # flange width, in
    tw_in:  float      # web thickness, in
    tf_in:  float      # flange thickness, in
    Fy_ksi: float = Fy_A992


@dataclass
class FrameNode:
    """Node in the FEM model."""
    id:         int
    xyz_in:     np.ndarray   # (3,) coordinates in inches
    restraints: np.ndarray   # (6,) bool — True = fixed DOF
    load:       np.ndarray   # (6,) applied nodal loads [Fx, Fy, Fz, Mx, My, Mz] kips/kip-in


@dataclass
class FrameElement:
    """3D beam-column element."""
    id:         int
    node_i:     int          # start node id
    node_j:     int          # end node id
    section:    FrameSection
    E_ksi:      float = E_STEEL
    G_ksi:      float = G_STEEL
    web_dir:    Optional[np.ndarray] = None   # local y-axis reference direction
    releases:   Optional[np.ndarray] = None   # (12,) bool — True = moment release at that DOF


@dataclass
class FEMResult:
    """Results from a single load case FEM analysis."""
    displacements:  np.ndarray     # (n_nodes, 6) global displacements [in, rad]
    reactions:      np.ndarray     # (n_nodes, 6) reaction forces at restrained DOFs
    element_forces: list[dict]     # per-element: {id, Pr, Vrx, Vry, Mrx, Mry, T}
    max_drift:      float          # maximum inter-story drift ratio
    max_deflection: float          # maximum vertical deflection, in
    natural_freq_hz: Optional[float] = None   # fundamental frequency (if mass provided)


class DirectStiffnessSolver:
    """
    Full 3D direct-stiffness frame solver.

    Usage
    -----
    >>> solver = DirectStiffnessSolver()
    >>> solver.add_node(0, [0, 0, 0], restraints=[1,1,1,1,1,1])
    >>> solver.add_node(1, [0, 0, 144], restraints=[0,0,0,0,0,0],
    ...                 load=[10, 0, -50, 0, 0, 0])
    >>> solver.add_element(0, 0, 1, section=my_W14x90)
    >>> result = solver.solve()
    >>> # Feed into code checks:
    >>> from steel_mesh_structural_rules import check_member_interaction
    >>> ef = result.element_forces[0]
    >>> check = check_member_interaction("COL-001",
    ...     Pr=ef['Pr'], Pc=phi_Pn, Mrx=ef['Mrx'], Mcx=phi_Mnx)
    """

    def __init__(self):
        self.nodes: dict[int, FrameNode] = {}
        self.elements: dict[int, FrameElement] = {}
        self._assembled = False

    # ── Node / element constructors ──────────────────────────────────────────

    def add_node(
        self,
        node_id: int,
        xyz_in: list | np.ndarray,
        restraints: list | np.ndarray | None = None,
        load: list | np.ndarray | None = None,
    ):
        """
        Add a node to the model.

        Parameters
        ----------
        node_id    : unique integer identifier
        xyz_in     : [x, y, z] coordinates in inches
        restraints : [rx, ry, rz, rmx, rmy, rmz]  1=fixed, 0=free (default: all free)
        load       : [Fx, Fy, Fz, Mx, My, Mz] applied nodal loads kips / kip-in
        """
        r = np.array(restraints if restraints is not None else [0]*6, dtype=bool)
        f = np.array(load if load is not None else [0.0]*6, dtype=float)
        self.nodes[node_id] = FrameNode(
            id=node_id,
            xyz_in=np.asarray(xyz_in, dtype=float),
            restraints=r,
            load=f,
        )

    def add_element(
        self,
        elem_id: int,
        node_i: int,
        node_j: int,
        section: FrameSection,
        E_ksi: float = E_STEEL,
        G_ksi: float = G_STEEL,
        web_dir: list | np.ndarray | None = None,
        releases: list | np.ndarray | None = None,
    ):
        """
        Add a 3D beam-column element.

        Parameters
        ----------
        elem_id : unique element identifier
        node_i  : start node id
        node_j  : end node id
        section : FrameSection with cross-section properties
        web_dir : reference vector for local y-axis orientation (default: auto)
        releases: (12,) bool array — True releases that DOF (moment release = hinge)
        """
        wd = np.asarray(web_dir, dtype=float) if web_dir is not None else None
        rl = np.asarray(releases, dtype=bool) if releases is not None else None
        self.elements[elem_id] = FrameElement(
            id=elem_id, node_i=node_i, node_j=node_j,
            section=section, E_ksi=E_ksi, G_ksi=G_ksi,
            web_dir=wd, releases=rl,
        )

    # ── Local stiffness matrix (12×12) ───────────────────────────────────────

    @staticmethod
    def _local_stiffness_12x12(
        L_in: float,
        A: float, Ix: float, Iy: float, J: float,
        E: float, G: float,
    ) -> np.ndarray:
        """
        Standard 12×12 Euler–Bernoulli beam element stiffness in local coords.

        Local axes:  x = along member, y = major bending, z = minor bending.

        DOF ordering per node: [ux, uy, uz, θx, θy, θz]
        Element DOF: [node_i(6), node_j(6)]

        Reference: McGuire et al., Table 4.1
        """
        L = L_in
        L2 = L * L
        L3 = L2 * L

        # Axial
        ea_L = E * A / L

        # Major-axis bending (Ix) — bending about local y, deflection in local z
        ei_z = E * Ix
        k_z1 = 12.0 * ei_z / L3
        k_z2 =  6.0 * ei_z / L2
        k_z3 =  4.0 * ei_z / L
        k_z4 =  2.0 * ei_z / L

        # Minor-axis bending (Iy) — bending about local z, deflection in local y
        ei_y = E * Iy
        k_y1 = 12.0 * ei_y / L3
        k_y2 =  6.0 * ei_y / L2
        k_y3 =  4.0 * ei_y / L
        k_y4 =  2.0 * ei_y / L

        # Torsion
        gj_L = G * J / L

        k = np.zeros((12, 12))

        # Row/col indices: 0=ux, 1=uy, 2=uz, 3=θx, 4=θy, 5=θz (node i)
        #                  6=ux, 7=uy, 8=uz, 9=θx, 10=θy, 11=θz (node j)

        # Axial: DOF 0, 6
        k[0, 0] = k[6, 6] =  ea_L
        k[0, 6] = k[6, 0] = -ea_L

        # Torsion: DOF 3, 9
        k[3, 3] = k[9, 9] =  gj_L
        k[3, 9] = k[9, 3] = -gj_L

        # Major bending (Iz plane — transverse z, rotation about y): DOF 2, 4, 8, 10
        k[2, 2]   =  k_z1;   k[2, 4]   =  k_z2;  k[2, 8]  = -k_z1;  k[2, 10]  =  k_z2
        k[4, 2]   =  k_z2;   k[4, 4]   =  k_z3;  k[4, 8]  = -k_z2;  k[4, 10]  =  k_z4
        k[8, 2]   = -k_z1;   k[8, 4]   = -k_z2;  k[8, 8]  =  k_z1;  k[8, 10]  = -k_z2
        k[10, 2]  =  k_z2;   k[10, 4]  =  k_z4;  k[10, 8] = -k_z2;  k[10, 10] =  k_z3

        # Minor bending (Iy plane — transverse y, rotation about z): DOF 1, 5, 7, 11
        k[1, 1]   =  k_y1;   k[1, 5]   = -k_y2;  k[1, 7]  = -k_y1;  k[1, 11]  = -k_y2
        k[5, 1]   = -k_y2;   k[5, 5]   =  k_y3;  k[5, 7]  =  k_y2;  k[5, 11]  =  k_y4
        k[7, 1]   = -k_y1;   k[7, 5]   =  k_y2;  k[7, 7]  =  k_y1;  k[7, 11]  =  k_y2
        k[11, 1]  = -k_y2;   k[11, 5]  =  k_y4;  k[11, 7] =  k_y2;  k[11, 11] =  k_y3

        return k

    # ── Rotation matrix ──────────────────────────────────────────────────────

    @staticmethod
    def _rotation_matrix(
        xi: np.ndarray,
        xj: np.ndarray,
        web_dir: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the 12×12 rotation matrix T for a 3D frame element.

        Local x-axis = element direction (i → j).
        Local y-axis = perpendicular to x in the plane defined by web_dir.
        Local z-axis = x × y (right-hand rule).

        Returns T such that k_global = T^T · k_local · T
        """
        dx = xj - xi
        L = np.linalg.norm(dx)
        x_local = dx / L

        # Default web direction: global Y if element is not parallel to Y
        if web_dir is None:
            if abs(x_local[1]) > 0.95:
                web_dir = np.array([0.0, 0.0, 1.0])
            else:
                web_dir = np.array([0.0, 1.0, 0.0])

        # Gram–Schmidt orthogonalization
        z_local = np.cross(x_local, web_dir)
        z_norm = np.linalg.norm(z_local)
        if z_norm < 1e-10:
            # web_dir parallel to element — fallback
            web_dir = np.array([1.0, 0.0, 0.0]) if abs(x_local[0]) < 0.95 else np.array([0.0, 0.0, 1.0])
            z_local = np.cross(x_local, web_dir)
            z_norm = np.linalg.norm(z_local)
        z_local /= z_norm
        y_local = np.cross(z_local, x_local)

        # 3×3 rotation
        R3 = np.array([x_local, y_local, z_local])

        # Expand to 12×12 block-diagonal
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*i+3, 3*i:3*i+3] = R3
        return T

    # ── Consistent mass matrix (optional, for modal analysis) ────────────────

    @staticmethod
    def _consistent_mass_12x12(
        L_in: float,
        A_in2: float,
        rho_ksi: float,   # mass density in kips·s²/in⁴ (= lb/in³ / 386.4)
    ) -> np.ndarray:
        """
        Consistent mass matrix for a uniform beam element.
        m = (ρAL/420) · [standard shape functions]

        Reference: Przemieniecki, "Theory of Matrix Structural Analysis", §11.3
        """
        m0 = rho_ksi * A_in2 * L_in / 420.0
        L = L_in

        m = np.zeros((12, 12))

        # Axial (DOF 0, 6)
        m[0, 0] = m[6, 6] = 140.0 * m0
        m[0, 6] = m[6, 0] =  70.0 * m0

        # Transverse z (DOF 2, 4, 8, 10) — major bending plane
        m[2, 2]   = 156*m0;   m[2, 4]   =  22*L*m0;  m[2, 8]   =  54*m0;  m[2, 10]  = -13*L*m0
        m[4, 2]   =  22*L*m0; m[4, 4]   =   4*L*L*m0;m[4, 8]   =  13*L*m0;m[4, 10]  = -3*L*L*m0
        m[8, 2]   =  54*m0;   m[8, 4]   =  13*L*m0;  m[8, 8]   = 156*m0;  m[8, 10]  = -22*L*m0
        m[10, 2]  = -13*L*m0; m[10, 4]  = -3*L*L*m0; m[10, 8]  = -22*L*m0;m[10, 10] =  4*L*L*m0

        # Transverse y (DOF 1, 5, 7, 11) — minor bending plane
        m[1, 1]   = 156*m0;   m[1, 5]   = -22*L*m0;  m[1, 7]   =  54*m0;  m[1, 11]  =  13*L*m0
        m[5, 1]   = -22*L*m0; m[5, 5]   =   4*L*L*m0;m[5, 7]   = -13*L*m0;m[5, 11]  = -3*L*L*m0
        m[7, 1]   =  54*m0;   m[7, 5]   = -13*L*m0;  m[7, 7]   = 156*m0;  m[7, 11]  =  22*L*m0
        m[11, 1]  =  13*L*m0; m[11, 5]  = -3*L*L*m0; m[11, 7]  =  22*L*m0;m[11, 11] =  4*L*L*m0

        # Torsion (DOF 3, 9) — lumped approximation
        Ip = A_in2  # approximate polar moment for mass
        mt = rho_ksi * Ip * L / 6.0
        m[3, 3] = m[9, 9] = 2 * mt
        m[3, 9] = m[9, 3] = mt

        return m

    # ── Assembly & solve ─────────────────────────────────────────────────────

    def solve(
        self,
        include_mass: bool = False,
        steel_density_pci: float = 0.284,  # lb/in³ for A992
        n_modes: int = 3,
    ) -> FEMResult:
        """
        Assemble the global stiffness (and optionally mass) matrix and solve.

        Steps
        -----
        1. Number DOFs (6 per node).
        2. Assemble global K (and M if requested) from element contributions.
        3. Apply boundary conditions via penalty method.
        4. Solve K·U = F for displacements.
        5. Back-calculate element forces in local coordinates.
        6. Optionally solve eigenvalue problem for natural frequencies.
        7. Compute inter-story drift and deflection metrics.

        Returns
        -------
        FEMResult with displacements, reactions, element forces, and metrics.
        """
        # ── DOF numbering ────────────────────────────────────────────────────
        sorted_nids = sorted(self.nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        n_nodes = len(sorted_nids)
        n_dof = 6 * n_nodes

        # ── Assembly (sparse triplets) ───────────────────────────────────────
        rows, cols, vals = [], [], []
        m_rows, m_cols, m_vals = [], [], []  # mass

        rho_ksi = steel_density_pci / 386.4   # convert lb/in³ → kips·s²/in⁴

        for eid, elem in self.elements.items():
            ni = self.nodes[elem.node_i]
            nj = self.nodes[elem.node_j]
            s = elem.section

            xi = ni.xyz_in
            xj = nj.xyz_in
            L_in = float(np.linalg.norm(xj - xi))
            if L_in < 1e-6:
                continue

            k_local = self._local_stiffness_12x12(
                L_in, s.A_in2, s.Ix_in4, s.Iy_in4, s.J_in4,
                elem.E_ksi, elem.G_ksi,
            )

            # Apply releases (moment hinges)
            if elem.releases is not None:
                for dof_idx in range(12):
                    if elem.releases[dof_idx]:
                        k_local[dof_idx, :] = 0.0
                        k_local[:, dof_idx] = 0.0

            T = self._rotation_matrix(xi, xj, elem.web_dir)
            k_global = T.T @ k_local @ T

            # Global DOF indices for this element
            ii = nid_to_idx[elem.node_i] * 6
            jj = nid_to_idx[elem.node_j] * 6
            dofs = list(range(ii, ii+6)) + list(range(jj, jj+6))

            for r in range(12):
                for c in range(12):
                    if abs(k_global[r, c]) > 1e-15:
                        rows.append(dofs[r])
                        cols.append(dofs[c])
                        vals.append(k_global[r, c])

            # Mass matrix
            if include_mass:
                m_local = self._consistent_mass_12x12(L_in, s.A_in2, rho_ksi)
                m_global = T.T @ m_local @ T
                for r in range(12):
                    for c in range(12):
                        if abs(m_global[r, c]) > 1e-15:
                            m_rows.append(dofs[r])
                            m_cols.append(dofs[c])
                            m_vals.append(m_global[r, c])

        K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsc()
        K = (K + K.T) / 2.0  # ensure symmetry

        # ── Load vector ──────────────────────────────────────────────────────
        F = np.zeros(n_dof)
        for nid in sorted_nids:
            node = self.nodes[nid]
            idx = nid_to_idx[nid] * 6
            F[idx:idx+6] = node.load

        # ── Boundary conditions (penalty method) ─────────────────────────────
        penalty = 1e12 * max(abs(K).max(), 1.0)
        restrained_dofs = []
        for nid in sorted_nids:
            node = self.nodes[nid]
            idx = nid_to_idx[nid] * 6
            for d in range(6):
                if node.restraints[d]:
                    restrained_dofs.append(idx + d)

        K_mod = K.tolil()
        for dof in restrained_dofs:
            K_mod[dof, dof] += penalty
            F[dof] = 0.0  # prescribed zero displacement
        K_mod = K_mod.tocsc()

        # ── Solve ────────────────────────────────────────────────────────────
        U = spsolve(K_mod, F)

        # ── Reactions ────────────────────────────────────────────────────────
        reactions = np.zeros((n_nodes, 6))
        R_full = K @ U - F   # residual at supports
        for nid in sorted_nids:
            node = self.nodes[nid]
            idx = nid_to_idx[nid] * 6
            for d in range(6):
                if node.restraints[d]:
                    reactions[nid_to_idx[nid], d] = R_full[idx + d]

        # ── Displacements reshaped ───────────────────────────────────────────
        displacements = U.reshape((n_nodes, 6))

        # ── Element forces (back-calculate in local coordinates) ─────────────
        element_forces = []
        for eid, elem in self.elements.items():
            ni = self.nodes[elem.node_i]
            nj = self.nodes[elem.node_j]
            s = elem.section
            xi, xj = ni.xyz_in, nj.xyz_in
            L_in = float(np.linalg.norm(xj - xi))
            if L_in < 1e-6:
                continue

            k_local = self._local_stiffness_12x12(
                L_in, s.A_in2, s.Ix_in4, s.Iy_in4, s.J_in4,
                elem.E_ksi, elem.G_ksi,
            )
            T = self._rotation_matrix(xi, xj, elem.web_dir)

            ii = nid_to_idx[elem.node_i] * 6
            jj = nid_to_idx[elem.node_j] * 6
            u_global = np.concatenate([U[ii:ii+6], U[jj:jj+6]])
            u_local = T @ u_global
            f_local = k_local @ u_local

            # Extract forces at node i end (conventional sign: positive = compression)
            # f_local: [Fx_i, Vy_i, Vz_i, Tx_i, My_i, Mz_i,
            #           Fx_j, Vy_j, Vz_j, Tx_j, My_j, Mz_j]
            Pr  = -f_local[0]       # axial: compression = -Fx_i
            Vry =  f_local[1]       # shear in local y
            Vrz =  f_local[2]       # shear in local z
            T_  =  f_local[3]       # torsion
            Mry =  f_local[4]       # moment about local y (major bending)
            Mrz =  f_local[5]       # moment about local z (minor bending)

            # Maximum moment along element (envelope: max of i-end and j-end)
            Mry_j = f_local[10]
            Mrz_j = f_local[11]
            Mrx_max = max(abs(Mry), abs(Mry_j)) / 12.0   # kip-in → kip-ft
            Mry_max = max(abs(Mrz), abs(Mrz_j)) / 12.0   # kip-in → kip-ft

            # Compute capacities for code checking
            KL_r_x = (L_in) / max(s.rx_in, 0.01)
            KL_r_y = (L_in) / max(s.ry_in, 0.01)
            KL_r   = max(KL_r_x, KL_r_y)
            Fcr    = column_fcr_aisc360_E3(KL_r, s.Fy_ksi, elem.E_ksi)
            phi_Pn = 0.90 * Fcr * s.A_in2           # kips
            phi_Mnx = 0.90 * s.Fy_ksi * s.Zx_in3 / 12.0  # kip-ft (assuming compact)
            phi_Mny = 0.90 * s.Fy_ksi * s.Zy_in3 / 12.0

            element_forces.append({
                'id':        eid,
                'Pr':        abs(Pr),            # kips (unsigned for DCR)
                'Pc':        phi_Pn,             # kips
                'Vrx':       Vry,                # kips
                'Vry':       Vrz,                # kips
                'Mrx':       Mrx_max,            # kip-ft
                'Mcx':       phi_Mnx,            # kip-ft
                'Mry':       Mry_max,            # kip-ft
                'Mcy':       phi_Mny,            # kip-ft
                'T':         T_,                 # kip-in
                'KL_r':      KL_r,               # dimensionless
                'L_in':      L_in,
                'DCR_H1':    aisc_360_interaction_H1(
                    abs(Pr), phi_Pn, Mrx_max, phi_Mnx, Mry_max, phi_Mny
                ),
            })

        # ── Story drift calculation ──────────────────────────────────────────
        # Group nodes by Z-coordinate, compute max lateral displacement diff
        z_coords = np.array([self.nodes[nid].xyz_in[2] for nid in sorted_nids])
        unique_z = np.unique(np.round(z_coords, 1))
        max_drift = 0.0
        for iz in range(1, len(unique_z)):
            z_lo, z_hi = unique_z[iz-1], unique_z[iz]
            h_story = z_hi - z_lo
            if h_story < 1.0:
                continue
            mask_lo = np.abs(z_coords - z_lo) < 1.0
            mask_hi = np.abs(z_coords - z_hi) < 1.0
            if mask_lo.any() and mask_hi.any():
                d_lo = np.max(np.linalg.norm(displacements[mask_lo, :2], axis=1))
                d_hi = np.max(np.linalg.norm(displacements[mask_hi, :2], axis=1))
                drift = abs(d_hi - d_lo) / h_story
                max_drift = max(max_drift, drift)

        # Max vertical deflection
        max_defl = float(np.max(np.abs(displacements[:, 2])))

        # ── Modal analysis (optional) ────────────────────────────────────────
        nat_freq = None
        if include_mass and m_vals:
            from scipy.sparse.linalg import eigsh
            M = sparse.coo_matrix(
                (m_vals, (m_rows, m_cols)), shape=(n_dof, n_dof)
            ).tocsc()
            M = (M + M.T) / 2.0

            # Apply same BCs to mass matrix
            M_mod = M.tolil()
            for dof in restrained_dofs:
                M_mod[dof, dof] += penalty * 1e-6
            M_mod = M_mod.tocsc()

            try:
                n_eff = min(n_modes, n_dof - len(restrained_dofs) - 1)
                if n_eff > 0:
                    eigenvalues, _ = eigsh(K_mod, k=n_eff, M=M_mod, which='SM')
                    omega_sq = eigenvalues[eigenvalues > 0]
                    if len(omega_sq) > 0:
                        nat_freq = float(np.sqrt(omega_sq[0]) / (2 * math.pi))
            except Exception:
                nat_freq = None

        return FEMResult(
            displacements=displacements,
            reactions=reactions,
            element_forces=element_forces,
            max_drift=max_drift,
            max_deflection=max_defl,
            natural_freq_hz=nat_freq,
        )


def run_code_checks_from_fem(
    fem_result: FEMResult,
    story_heights_in: list[float] | None = None,
) -> list[CodeCheckResult]:
    """
    Bridge function: takes FEM results and runs all applicable code checks
    from steel_mesh_structural_rules Section 11.

    Returns a list of CodeCheckResult for every element and every story.
    """
    checks = []

    # ── Per-element checks ───────────────────────────────────────────────────
    for ef in fem_result.element_forces:
        eid_str = f"ELEM-{ef['id']:04d}"

        # H1 interaction
        checks.append(check_member_interaction(
            eid_str,
            Pr=ef['Pr'], Pc=ef['Pc'],
            Mrx=ef['Mrx'], Mcx=ef['Mcx'],
            Mry=ef['Mry'], Mcy=ef['Mcy'],
        ))

        # Slenderness
        checks.append(check_slenderness(eid_str, ef['KL_r']))

        # Beam deflection (use element length as span)
        span_m = ef['L_in'] * 0.0254
        delta_m = fem_result.max_deflection * 0.0254
        checks.append(check_beam_deflection(eid_str, delta_m, span_m))

    # ── Story drift check ────────────────────────────────────────────────────
    if story_heights_in is not None:
        for i, h_in in enumerate(story_heights_in):
            h_m = h_in * 0.0254
            drift_ratio = fem_result.max_drift
            delta_m = drift_ratio * h_m
            checks.append(check_story_drift(f"STORY-{i+1:02d}", delta_m, h_m))
    elif fem_result.max_drift > 0:
        checks.append(check_story_drift(
            "STORY-MAX", fem_result.max_drift * 4.0 * 39.37 * 0.0254, 4.0
        ))

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# MODULE B — COTANGENT-WEIGHT LAPLACE–BELTRAMI CURVATURE OPERATOR
# ══════════════════════════════════════════════════════════════════════════════
#
# Theory: The discrete Laplace–Beltrami operator Δ_S applied to vertex
# positions gives the mean curvature normal:
#
#   Δ_S f(v_i) = (1 / 2A_i) Σ_{j∈N(i)} (cot α_ij + cot β_ij)(f(v_j) - f(v_i))
#
# where α_ij and β_ij are the angles opposite edge (i,j) in the two
# adjacent triangles, and A_i is the mixed Voronoi area at vertex i.
#
# The mean curvature at vertex i:
#   H_i = ‖Δ_S v_i‖ / 2   (with sign from dot product with normal)
#
# Gaussian curvature via the angle defect (Gauss–Bonnet):
#   K_i = (2π − Σ_f θ_i^f) / A_i
#
# References:
#   - Meyer, Desbrun, Schröder, Barr (2003), "Discrete Differential-Geometry
#     Operators for Triangulated 2-Manifolds", VisMath
#   - Botsch, Kobbelt, Pauly, Alliez, Lévy (2010), "Polygon Mesh Processing"
# ══════════════════════════════════════════════════════════════════════════════


def cotangent_laplace_beltrami(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """
    Build the cotangent-weight Laplace–Beltrami operator L and the
    mixed Voronoi area vector A for a triangle mesh.

    Parameters
    ----------
    vertices : (N, 3) vertex positions
    faces    : (M, 3) triangle face indices

    Returns
    -------
    L : (N, N) sparse cotangent Laplacian matrix
    A : (N,)   mixed Voronoi area per vertex
    """
    N = len(vertices)
    M = len(faces)

    ii_list, jj_list, ww_list = [], [], []
    A_mixed = np.zeros(N)

    for f_idx in range(M):
        i, j, k = faces[f_idx]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # Edge vectors
        eij = vj - vi;  eik = vk - vi
        ejk = vk - vj;  eji = vi - vj
        eki = vi - vk;  ekj = vj - vk

        # Face area
        cross = np.cross(eij, eik)
        area2 = np.linalg.norm(cross)
        face_area = area2 / 2.0

        if face_area < 1e-15:
            continue

        # Cotangents of each angle
        cot_i = np.dot(eij, eik) / max(np.linalg.norm(np.cross(eij, eik)), 1e-15)
        cot_j = np.dot(eji, ejk) / max(np.linalg.norm(np.cross(eji, ejk)), 1e-15)
        cot_k = np.dot(eki, ekj) / max(np.linalg.norm(np.cross(eki, ekj)), 1e-15)

        # Clamp cotangents to avoid numerical issues
        cot_i = max(min(cot_i, 1e6), -1e6)
        cot_j = max(min(cot_j, 1e6), -1e6)
        cot_k = max(min(cot_k, 1e6), -1e6)

        # Cotangent weights for each edge:
        # Edge (j,k): opposite vertex i → weight = cot_i
        # Edge (i,k): opposite vertex j → weight = cot_j
        # Edge (i,j): opposite vertex k → weight = cot_k
        pairs_weights = [
            (j, k, cot_i),
            (i, k, cot_j),
            (i, j, cot_k),
        ]

        for (a, b, w) in pairs_weights:
            ii_list.extend([a, b, a, b])
            jj_list.extend([b, a, a, b])
            ww_list.extend([w/2, w/2, -w/2, -w/2])

        # Mixed Voronoi area allocation
        dot_i = np.dot(eij, eik)
        dot_j = np.dot(eji, ejk)
        dot_k = np.dot(eki, ekj)

        if dot_i < 0:
            A_mixed[i] += face_area / 2.0
            A_mixed[j] += face_area / 4.0
            A_mixed[k] += face_area / 4.0
        elif dot_j < 0:
            A_mixed[j] += face_area / 2.0
            A_mixed[i] += face_area / 4.0
            A_mixed[k] += face_area / 4.0
        elif dot_k < 0:
            A_mixed[k] += face_area / 2.0
            A_mixed[i] += face_area / 4.0
            A_mixed[j] += face_area / 4.0
        else:
            # Non-obtuse → true Voronoi areas
            A_mixed[i] += (np.dot(eij, eij) * cot_k + np.dot(eik, eik) * cot_j) / 8.0
            A_mixed[j] += (np.dot(eji, eji) * cot_k + np.dot(ejk, ejk) * cot_i) / 8.0
            A_mixed[k] += (np.dot(eki, eki) * cot_j + np.dot(ekj, ekj) * cot_i) / 8.0

    L = sparse.coo_matrix(
        (ww_list, (ii_list, jj_list)), shape=(N, N),
    ).tocsc()

    A_mixed = np.maximum(A_mixed, 1e-12)
    return L, A_mixed


def compute_mean_curvature_vertex(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-vertex mean curvature H using the cotangent Laplace–Beltrami
    operator.

    H_i = (1/2) ‖Δ_S v_i‖ / A_i   (signed by dot with normal)

    Returns
    -------
    H : (N,) signed mean curvature per vertex (1/m if input in metres)
    """
    N = len(vertices)
    L, A = cotangent_laplace_beltrami(vertices, faces)

    # Laplacian of position
    Hn = np.zeros((N, 3))
    for dim in range(3):
        Hn[:, dim] = L @ vertices[:, dim]

    inv_2A = 1.0 / (2.0 * A)
    Hn_normalized = Hn * inv_2A[:, None]

    H_unsigned = np.linalg.norm(Hn_normalized, axis=1)

    if vertex_normals is None:
        vertex_normals = _compute_vertex_normals(vertices, faces)
    sign = np.sign(np.sum(Hn_normalized * vertex_normals, axis=1))
    sign[sign == 0] = 1.0

    return sign * H_unsigned


def compute_gaussian_curvature_vertex(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute per-vertex Gaussian curvature K via the angle defect
    (discrete Gauss–Bonnet theorem).

    K_i = (2π − Σ_{f∈star(i)} θ_i^f) / A_mixed_i

    Returns
    -------
    K : (N,) Gaussian curvature per vertex (1/m² if input in metres)
    """
    N = len(vertices)
    _, A = cotangent_laplace_beltrami(vertices, faces)

    angle_sum = np.zeros(N)

    for f_idx in range(len(faces)):
        i, j, k = faces[f_idx]

        for (a, b, c) in [(i, j, k), (j, k, i), (k, i, j)]:
            ea = vertices[b] - vertices[a]
            eb = vertices[c] - vertices[a]
            denom = np.linalg.norm(ea) * np.linalg.norm(eb)
            if denom > 1e-15:
                cos_angle = np.clip(np.dot(ea, eb) / denom, -1.0, 1.0)
                angle_sum[a] += math.acos(cos_angle)

    K = (2.0 * math.pi - angle_sum) / A
    return K


def compute_face_curvatures_from_vertex(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-face mean and Gaussian curvature by averaging vertex values.

    This is the production replacement for _estimate_face_mean_curvature and
    _estimate_face_gaussian_curvature in the original module.

    Returns
    -------
    face_H : (M,) mean curvature per face
    face_K : (M,) Gaussian curvature per face
    """
    H_vertex = compute_mean_curvature_vertex(vertices, faces)
    K_vertex = compute_gaussian_curvature_vertex(vertices, faces)

    face_H = (H_vertex[faces[:, 0]] + H_vertex[faces[:, 1]] + H_vertex[faces[:, 2]]) / 3.0
    face_K = (K_vertex[faces[:, 0]] + K_vertex[faces[:, 1]] + K_vertex[faces[:, 2]]) / 3.0

    return face_H, face_K


def classify_curvature(
    face_H: np.ndarray,
    face_K: np.ndarray,
    synclastic_threshold: float = 0.01,
) -> str:
    """
    Classify mesh curvature type from per-face H and K values.

    K > 0 (everywhere)  → synclastic (dome, sphere)
    K < 0 (everywhere)  → anticlastic (saddle, hyperboloid)
    K ≈ 0 (everywhere)  → flat or cylindrical
    mixed K signs        → compound curvature

    Returns: 'synclastic' | 'anticlastic' | 'flat' | 'compound'
    """
    pos_K = (face_K > synclastic_threshold).sum()
    neg_K = (face_K < -synclastic_threshold).sum()
    total = len(face_K)

    if pos_K > 0 and neg_K > 0:
        return "compound"
    elif pos_K > neg_K and pos_K > 0.1 * total:
        return "synclastic"
    elif neg_K > pos_K and neg_K > 0.1 * total:
        return "anticlastic"
    else:
        return "flat"


def _compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Area-weighted vertex normals from face normals."""
    N = len(vertices)
    normals = np.zeros((N, 3))
    for f_idx in range(len(faces)):
        i, j, k = faces[f_idx]
        e1 = vertices[j] - vertices[i]
        e2 = vertices[k] - vertices[i]
        fn = np.cross(e1, e2)
        normals[i] += fn
        normals[j] += fn
        normals[k] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return normals / norms


# ══════════════════════════════════════════════════════════════════════════════
# MODULE C — MESH SURFACE PROJECTION FOR DIAGRID LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
#
# Problem: The parametric diagrid skeleton from generate_diagrid_layout places
# nodes on a rectangular bounding box.  For arbitrary curved meshes, these
# nodes must be projected onto the actual mesh surface.
#
# Algorithm:
# 1. For each parametric node, find the closest triangle on the mesh surface.
# 2. Compute the closest point on that triangle (barycentric projection).
# 3. Optionally smooth the result with Laplacian relaxation to even out
#    spacing on highly curved regions.
#
# References:
#   - Ericson (2005), "Real-Time Collision Detection", Ch. 5
#   - Botsch et al. (2010), "Polygon Mesh Processing", Ch. 4
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectedDiagridLayout(DiagridLayout):
    """Extended DiagridLayout with mesh-projected node positions."""
    projected_nodes:   list[DiagridNode] = field(default_factory=list)
    projection_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    surface_normals:   np.ndarray = field(default_factory=lambda: np.array([]))


def closest_point_on_triangle(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the closest point on triangle (v0, v1, v2) to point p.
    Returns (closest_point, barycentric_coords).

    Uses the Voronoi region method (Ericson 2005, §5.1.5).
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = p - v0

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return v0.copy(), np.array([1.0, 0.0, 0.0])

    bp = p - v1
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return v1.copy(), np.array([0.0, 1.0, 0.0])

    cp = p - v2
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return v2.copy(), np.array([0.0, 0.0, 1.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v_ = d1 / (d1 - d3)
        return v0 + v_ * ab, np.array([1.0 - v_, v_, 0.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return v0 + w * ac, np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return v1 + w * (v2 - v1), np.array([0.0, 1.0 - w, w])

    denom = 1.0 / (va + vb + vc)
    v_ = vb * denom
    w_ = vc * denom
    return v0 + ab * v_ + ac * w_, np.array([1.0 - v_ - w_, v_, w_])


def project_points_to_mesh(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project an array of points onto a triangle mesh surface.

    For each point, finds the closest point on the mesh surface using
    brute-force closest-triangle search with centroid pre-filter.

    Parameters
    ----------
    points   : (P, 3) points to project
    vertices : (N, 3) mesh vertices
    faces    : (M, 3) mesh face indices

    Returns
    -------
    projected     : (P, 3) closest points on the mesh surface
    face_indices  : (P,)   index of the face each point was projected to
    bary_coords   : (P, 3) barycentric coordinates on the respective face
    """
    P = len(points)
    M = len(faces)

    projected = np.zeros((P, 3))
    face_indices = np.zeros(P, dtype=int)
    bary_coords = np.zeros((P, 3))

    f_v0 = vertices[faces[:, 0]]
    f_v1 = vertices[faces[:, 1]]
    f_v2 = vertices[faces[:, 2]]
    face_centroids = (f_v0 + f_v1 + f_v2) / 3.0

    for pi in range(P):
        p = points[pi]

        dists_to_centroids = np.linalg.norm(face_centroids - p, axis=1)
        n_candidates = min(50, M)
        if n_candidates >= M:
            candidates = np.arange(M)
        else:
            candidates = np.argpartition(dists_to_centroids, n_candidates)[:n_candidates]

        best_dist = np.inf
        best_pt = p.copy()
        best_face = 0
        best_bary = np.array([1.0/3, 1.0/3, 1.0/3])

        for fi in candidates:
            cp, bc = closest_point_on_triangle(p, f_v0[fi], f_v1[fi], f_v2[fi])
            d = np.linalg.norm(cp - p)
            if d < best_dist:
                best_dist = d
                best_pt = cp
                best_face = fi
                best_bary = bc

        projected[pi] = best_pt
        face_indices[pi] = best_face
        bary_coords[pi] = best_bary

    return projected, face_indices, bary_coords


def laplacian_smooth_on_surface(
    points: np.ndarray,
    adjacency: list[list[int]],
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Laplacian smoothing of projected points constrained to the mesh surface.

    Each iteration:
    1. Move each point toward the centroid of its neighbors.
    2. Re-project onto the mesh surface.

    Parameters
    ----------
    points     : (P, 3) initial positions on the mesh surface
    adjacency  : list of neighbor index lists for each point
    vertices   : (N, 3) mesh vertices
    faces      : (M, 3) mesh faces
    iterations : number of smoothing iterations
    alpha      : blending factor (0 = no smoothing, 1 = full Laplacian)

    Returns
    -------
    smoothed : (P, 3) smoothed positions on the mesh surface
    """
    pts = points.copy()
    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(len(pts)):
            if len(adjacency[i]) == 0:
                continue
            neighbors = pts[adjacency[i]]
            centroid = neighbors.mean(axis=0)
            new_pts[i] = (1 - alpha) * pts[i] + alpha * centroid
        pts, _, _ = project_points_to_mesh(new_pts, vertices, faces)
    return pts


def generate_diagrid_layout_projected(
    desc: MeshDescriptors,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    floor_height_m: float = 4.0,
    smooth_iterations: int = 3,
) -> ProjectedDiagridLayout:
    """
    Full diagrid layout generation with mesh surface projection.

    This replaces the original generate_diagrid_layout for curved meshes.

    Steps
    -----
    1. Generate the parametric diagrid skeleton (rectangular approximation).
    2. For each perimeter node, project onto the mesh surface using
       closest-point projection.
    3. Apply Laplacian smoothing (surface-constrained) to regularize spacing.
    4. Recompute member lengths and angles from projected positions.
    5. Return the ProjectedDiagridLayout with both original and projected nodes.
    """
    theta  = diagrid_theta_from_HW(desc.H_W_ratio)
    H_m    = diagrid_module_height_m(floor_height_m, desc.stories_per_module)
    perim  = 2 * (desc.width_m + desc.depth_m)
    n_bays = diagrid_bay_count(perim, H_m, theta)
    L_diag = diagrid_diagonal_length_m(H_m, theta)

    # Section family
    if desc.height_m < 30:
        sec_fam, node_tp = "HSS8-12", "plate_welded"
    elif desc.height_m < 90:
        sec_fam, node_tp = "HSS10-16", "plate_welded_or_cast"
    elif desc.height_m < 180:
        sec_fam, node_tp = "buildup_box_W14", "cast_steel"
    else:
        sec_fam, node_tp = "buildup_box_mega", "cast_steel_AESS4"

    # ── Generate parametric nodes ────────────────────────────────────────────
    n_modules_z = max(1, int(desc.story_count / desc.stories_per_module))
    bay_width = perim / n_bays

    parametric_points = []
    node_floor_flags = []

    for iz in range(n_modules_z + 1):
        z = iz * H_m + desc.bbox_min[2]
        for ibay in range(n_bays):
            x_offset = bay_width * ibay + (bay_width / 2.0 if iz % 2 == 0 else 0.0)
            cx, cy = _perimeter_xy(x_offset, perim, desc)
            parametric_points.append([cx, cy, z])
            node_floor_flags.append(True)

    parametric_points = np.array(parametric_points)

    # ── Project onto mesh surface ────────────────────────────────────────────
    projected_pts, face_ids, bary = project_points_to_mesh(
        parametric_points, mesh_vertices, mesh_faces
    )

    # Preserve Z-coordinate (structural floor levels)
    for pi in range(len(projected_pts)):
        orig_z = parametric_points[pi, 2]
        if abs(projected_pts[pi, 2] - orig_z) > H_m * 0.5:
            projected_pts[pi, 2] = orig_z

    # ── Build adjacency for smoothing ────────────────────────────────────────
    adjacency: list[list[int]] = [[] for _ in range(len(projected_pts))]
    for iz in range(n_modules_z + 1):
        base = iz * n_bays
        for ibay in range(n_bays):
            idx = base + ibay
            left = base + (ibay - 1) % n_bays
            right = base + (ibay + 1) % n_bays
            adjacency[idx].extend([left, right])
            if iz > 0:
                adjacency[idx].append((iz - 1) * n_bays + ibay)
            if iz < n_modules_z:
                adjacency[idx].append((iz + 1) * n_bays + ibay)

    # ── Laplacian smoothing on surface ───────────────────────────────────────
    if smooth_iterations > 0:
        projected_pts = laplacian_smooth_on_surface(
            projected_pts, adjacency,
            mesh_vertices, mesh_faces,
            iterations=smooth_iterations, alpha=0.4,
        )

    # ── Compute surface normals at projected points ──────────────────────────
    face_normals = np.cross(
        mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]],
        mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 0]],
    )
    fn_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(fn_norms, 1e-15)
    surface_normals = face_normals[face_ids]

    # ── Build node and member lists ──────────────────────────────────────────
    nodes_param = []
    nodes_proj  = []
    for pi in range(len(parametric_points)):
        nid = f"DG_{pi:04d}"
        nodes_param.append(DiagridNode(
            id=nid, xyz=parametric_points[pi], floor_level=node_floor_flags[pi]
        ))
        nodes_proj.append(DiagridNode(
            id=nid, xyz=projected_pts[pi], floor_level=node_floor_flags[pi]
        ))

    # Connect diagonals + ring beams
    members = []
    member_id = 0
    for iz in range(n_modules_z):
        base_lo = iz * n_bays
        base_hi = (iz + 1) * n_bays
        for ibay in range(n_bays):
            ni_idx = base_lo + ibay
            for offset in [0, 1]:
                nj_idx = base_hi + (ibay + offset) % n_bays
                if nj_idx >= len(nodes_proj):
                    continue
                ni_pos = projected_pts[ni_idx]
                nj_pos = projected_pts[nj_idx]
                length = float(np.linalg.norm(nj_pos - ni_pos))
                dz = abs(nj_pos[2] - ni_pos[2])
                dh = math.sqrt(max(length**2 - dz**2, 0))
                actual_theta = math.degrees(math.atan2(dz, max(dh, 1e-6)))

                members.append(DiagridMember(
                    id=f"DG_MEM_{member_id:04d}",
                    node_i=nodes_proj[ni_idx].id,
                    node_j=nodes_proj[nj_idx].id,
                    theta_deg=actual_theta,
                    length_m=length,
                    role="diagonal",
                ))
                member_id += 1

        # Ring beams at each level
        for ibay in range(n_bays):
            ni_idx = base_hi + ibay
            nj_idx = base_hi + (ibay + 1) % n_bays
            if nj_idx >= len(nodes_proj):
                continue
            length = float(np.linalg.norm(
                projected_pts[nj_idx] - projected_pts[ni_idx]
            ))
            members.append(DiagridMember(
                id=f"DG_MEM_{member_id:04d}",
                node_i=nodes_proj[ni_idx].id,
                node_j=nodes_proj[nj_idx].id,
                theta_deg=0.0,
                length_m=length,
                role="ring_beam",
            ))
            member_id += 1

    proj_errors = np.linalg.norm(projected_pts - parametric_points, axis=1)

    return ProjectedDiagridLayout(
        nodes=nodes_param,
        members=members,
        theta_deg=theta,
        module_h_m=H_m,
        bay_count=n_bays,
        section_family=sec_fam,
        node_type=node_tp,
        projected_nodes=nodes_proj,
        projection_errors=proj_errors,
        surface_normals=surface_normals,
    )


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — UPGRADED extract_mesh_descriptors USING MODULE B
# ══════════════════════════════════════════════════════════════════════════════

def extract_mesh_descriptors_v2(
    vertices: np.ndarray,
    faces: np.ndarray,
    creases: np.ndarray,
    floor_height_m: float = 4.0,
    curvature_flat_threshold: float = 0.05,
    void_threshold: float = 0.20,
) -> MeshDescriptors:
    """
    Drop-in replacement for extract_mesh_descriptors that uses the
    cotangent-weight Laplace–Beltrami curvature operator (Module B)
    instead of the placeholder estimators.

    Same interface, same return type — just better curvature values.
    """
    from steel_mesh_structural_rules import (
        _signed_mesh_volume, _detect_shaft_candidates,
    )

    verts = np.asarray(vertices, dtype=float)
    fcs   = np.asarray(faces, dtype=int)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    centroid = verts.mean(axis=0)
    dims     = bbox_max - bbox_min
    height_m = float(dims[2])
    width_m  = float(min(dims[0], dims[1]))
    depth_m  = float(max(dims[0], dims[1]))
    H_W      = height_m / max(width_m, 1e-6)

    bbox_vol = float(dims[0] * dims[1] * dims[2])
    mesh_vol = _signed_mesh_volume(verts, fcs)
    void_frac = max(0.0, 1.0 - abs(mesh_vol) / max(bbox_vol, 1e-6))

    v0 = verts[fcs[:, 0]]; v1 = verts[fcs[:, 1]]; v2 = verts[fcs[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) * 0.5
    surface_area = float(face_areas.sum())
    face_centroids = (v0 + v1 + v2) / 3.0

    # ── CURVATURE (cotangent Laplace–Beltrami) ───────────────────────────────
    face_H, face_K = compute_face_curvatures_from_vertex(verts, fcs)
    max_mean_curv  = float(np.abs(face_H).max()) if len(face_H) else 0.0
    max_gauss_curv = float(np.abs(face_K).max()) if len(face_K) else 0.0
    curvature_type = classify_curvature(face_H, face_K)

    flat_mask  = np.abs(face_H) < curvature_flat_threshold
    flat_faces = [np.where(flat_mask)[0]]

    shaft_candidates = _detect_shaft_candidates(verts, fcs, face_centroids, centroid)
    reentrant = [np.where(face_K < -0.05)[0]]

    z_base_thresh = bbox_min[2] + 0.05 * height_m
    support_fp = np.where(verts[:, 2] <= z_base_thresh)[0]

    story_count = max(1, round(height_m / floor_height_m))
    if H_W > 8:     aspect_cat = "super_tall"
    elif H_W > 4:   aspect_cat = "tall"
    elif H_W > 1:   aspect_cat = "mid_rise"
    else:           aspect_cat = "low_wide"

    if story_count <= 10:     stories_per_module = 2
    elif story_count <= 20:   stories_per_module = 3
    else:                     stories_per_module = 4

    return MeshDescriptors(
        centroid=centroid, bbox_min=bbox_min, bbox_max=bbox_max,
        height_m=height_m, width_m=width_m, depth_m=depth_m,
        H_W_ratio=H_W,
        volume_m3=abs(mesh_vol), surface_area_m2=surface_area,
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


# ══════════════════════════════════════════════════════════════════════════════
# UPGRADED PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline_v2(
    vertices: np.ndarray,
    faces: np.ndarray,
    creases: np.ndarray,
    floor_height_m: float = 4.0,
    sdc: str = "D",
    program: str = "office",
    sdl_psf: float = 20.0,
    run_fem: bool = False,
    sections: dict[str, FrameSection] | None = None,
) -> dict:
    """
    Upgraded end-to-end pipeline using all three modules:

    1. Extract mesh descriptors (cotangent LB curvature — Module B)
    2. Select structural system
    3. Compute gravity loads
    4. Generate diagrid layout with mesh projection (Module C)
    5. Optionally run FEM analysis (Module A)
    6. Run code checks from FEM results
    7. Return consolidated report dict
    """
    from steel_mesh_structural_rules import (
        select_structural_system, gravity_loads_psf,
    )

    desc = extract_mesh_descriptors_v2(vertices, faces, creases, floor_height_m)
    system = select_structural_system(desc, sdc=sdc, program=program)
    loads = gravity_loads_psf(program, sdl_psf)

    diagrid = None
    if "DIAGRID" in system.lateral_system:
        diagrid = generate_diagrid_layout_projected(
            desc, vertices, faces, floor_height_m
        )

    face_H, face_K = compute_face_curvatures_from_vertex(vertices, faces)
    curvature_data = {
        'face_mean_curvature': face_H,
        'face_gaussian_curvature': face_K,
        'curvature_type': desc.curvature_type,
        'max_H': desc.max_mean_curvature,
        'max_K': desc.max_gaussian_curv,
    }

    fem_result = None
    checks = []
    if run_fem and diagrid is not None and sections is not None:
        solver = DirectStiffnessSolver()
        node_map = {}
        nodes_to_use = diagrid.projected_nodes or diagrid.nodes
        for i, node in enumerate(nodes_to_use):
            xyz_in = node.xyz * 39.3701
            is_base = (node.xyz[2] - desc.bbox_min[2]) < 0.1
            restraints = [1,1,1,1,1,1] if is_base else [0,0,0,0,0,0]
            solver.add_node(i, xyz_in, restraints=restraints)
            node_map[node.id] = i

        default_sec = list(sections.values())[0] if sections else None
        for mi, member in enumerate(diagrid.members):
            if member.node_i in node_map and member.node_j in node_map:
                sec = sections.get(member.role, default_sec)
                if sec is not None:
                    solver.add_element(
                        mi, node_map[member.node_i], node_map[member.node_j], sec
                    )

        total_load_psf = loads['SDL_psf'] + loads['L_psf']
        bay_area_ft2 = (desc.width_m * desc.depth_m * 10.764) / max(len(node_map), 1)
        P_per_node = total_load_psf * bay_area_ft2 / 1000.0

        for nid_str, nid_int in node_map.items():
            node = solver.nodes[nid_int]
            if not node.restraints[2]:
                node.load[2] = -P_per_node

        fem_result = solver.solve(include_mass=True)
        checks = run_code_checks_from_fem(fem_result)

    return {
        'descriptors':  desc,
        'system':       system,
        'loads':        loads,
        'diagrid':      diagrid,
        'curvature':    curvature_data,
        'fem_result':   fem_result,
        'checks':       checks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("FEM SOLVER & ADVANCED GEOMETRY — INTEGRATION TEST")
    print("=" * 80)

    # ── Test mesh: 30m × 30m × 120m box ─────────────────────────────────────
    W, D, H = 30.0, 30.0, 120.0
    verts = np.array([
        [0, 0, 0],  [W, 0, 0],  [W, D, 0],  [0, D, 0],
        [0, 0, H],  [W, 0, H],  [W, D, H],  [0, D, H],
    ], dtype=float)
    fcs = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [1,2,6],[1,6,5],
        [2,3,7],[2,7,6],
        [3,0,4],[3,4,7],
    ], dtype=int)
    creases = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4]], dtype=int)

    # ── Module B test: Curvature ─────────────────────────────────────────────
    print("\n── Module B: Cotangent Laplace–Beltrami Curvature ──")
    face_H, face_K = compute_face_curvatures_from_vertex(verts, fcs)
    print(f"  Mean curvature range:     [{face_H.min():.4f}, {face_H.max():.4f}] 1/m")
    print(f"  Gaussian curvature range: [{face_K.min():.4f}, {face_K.max():.4f}] 1/m²")
    print(f"  Curvature type: {classify_curvature(face_H, face_K)}")

    # ── Module C test: Mesh projection ───────────────────────────────────────
    print("\n── Module C: Mesh Surface Projection ──")
    desc = extract_mesh_descriptors_v2(verts, fcs, creases)
    print(f"  H/W = {desc.H_W_ratio:.2f}  →  {desc.aspect_category}")
    print(f"  Curvature type: {desc.curvature_type}")

    layout = generate_diagrid_layout_projected(desc, verts, fcs, floor_height_m=4.0)
    print(f"  Diagrid: θ={layout.theta_deg}°, {layout.bay_count} bays, "
          f"{len(layout.members)} members, {len(layout.nodes)} nodes")
    print(f"  Projection errors: mean={layout.projection_errors.mean():.3f}m, "
          f"max={layout.projection_errors.max():.3f}m")

    # ── Module A test: Simple FEM ────────────────────────────────────────────
    print("\n── Module A: Direct Stiffness FEM Solver ──")
    solver = DirectStiffnessSolver()

    # Simple cantilevered column test: W14×90
    w14x90 = FrameSection(
        name="W14X90", A_in2=26.5, Ix_in4=999.0, Iy_in4=362.0,
        J_in4=4.06, Zx_in3=157.0, Zy_in3=83.3,
        rx_in=6.14, ry_in=3.70, d_in=14.0, bf_in=14.5,
        tw_in=0.440, tf_in=0.710, Fy_ksi=50.0,
    )

    # Cantilever: 12ft column, fixed base, 100-kip lateral + 500-kip gravity
    solver.add_node(0, [0, 0, 0], restraints=[1,1,1,1,1,1])
    solver.add_node(1, [0, 0, 144], load=[100, 0, -500, 0, 0, 0])
    solver.add_element(0, 0, 1, w14x90)

    result = solver.solve(include_mass=True)
    print(f"  Tip displacement: dx={result.displacements[1, 0]:.4f} in, "
          f"dz={result.displacements[1, 2]:.6f} in")
    print(f"  Max drift ratio: {result.max_drift:.6f}")

    ef = result.element_forces[0]
    print(f"  Element forces: Pr={ef['Pr']:.1f} kips, Mrx={ef['Mrx']:.1f} kip-ft")
    print(f"  Capacities:     Pc={ef['Pc']:.1f} kips, Mcx={ef['Mcx']:.1f} kip-ft")
    print(f"  DCR (H1):       {ef['DCR_H1']:.3f}")

    # Run code checks
    checks = run_code_checks_from_fem(result)
    print(f"\n  Code checks ({len(checks)} total):")
    for c in checks:
        print(f"    [{c.status.upper():4s}] {c.check_type}: {c.message}")

    print("\n" + "=" * 80)
    print("ALL INTEGRATION TESTS COMPLETE")
    print("=" * 80)
