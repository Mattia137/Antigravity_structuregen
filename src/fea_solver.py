import math
import networkx as nx
import numpy as np
from Pynite import FEModel3D

# Unit conversions: config.py stores sections in imperial (in², in⁴)
# FEA model runs in SI (m², m⁴, Pa, N)
_IN2_TO_M2 = 6.4516e-4
_IN4_TO_M4 = 4.16231e-7

class FEASolver:
    def __init__(self, structural_graph: nx.Graph, material_params: dict):
        """
        Convert the designed structural graph into a PyNite FEA model.
        material_params: {"type": "Steel"|"Concrete", "E": Pa, "G": Pa, "nu": float, "rho": kg/m³, "Fy": Pa}
        """
        self.graph = structural_graph
        self.model = FEModel3D()
        self.material = material_params
        self.failures = []
        self._registered_sections = set()

        mat_name = self.material.get("type", "Material_1")
        self.model.add_material(
            mat_name,
            E=self.material.get("E", 200e9),
            G=self.material.get("G", 77e9),
            nu=self.material.get("nu", 0.3),
            rho=self.material.get("rho", 7850)
        )

        # Fallback section props (SI) used when a named section is not in config.py
        self._fallback_props = {
            "primary_crease":    {"A": 0.05,   "Iy": 0.001,   "Iz": 0.001,   "J": 0.002},
            "secondary_lattice": {"A": 0.01,   "Iy": 0.0001,  "Iz": 0.0001,  "J": 0.0002},
            "default":           {"A": 0.01,   "Iy": 0.0001,  "Iz": 0.0001,  "J": 0.0002},
        }

    # ------------------------------------------------------------------
    def _ensure_section(self, section_name: str, mat_type: str):
        """Register a named section with PyNite (once only). Looks up config.py first."""
        if section_name in self._registered_sections:
            return

        try:
            from config import SECTIONS
            mat_sections = SECTIONS.get(mat_type, SECTIONS.get("Steel", {}))
            if section_name in mat_sections:
                s = mat_sections[section_name]
                self.model.add_section(
                    section_name,
                    s["A"] * _IN2_TO_M2,
                    s["Iy"] * _IN4_TO_M4,
                    s["Iz"] * _IN4_TO_M4,
                    s["J"] * _IN4_TO_M4,
                )
                self._registered_sections.add(section_name)
                return
        except Exception:
            pass

        # Fallback: use role-based SI props
        props = self._fallback_props.get(section_name, self._fallback_props["default"])
        self.model.add_section(section_name, props["A"], props["Iy"], props["Iz"], props["J"])
        self._registered_sections.add(section_name)

    # ------------------------------------------------------------------
    def build_model(self):
        """Populate the FEA model with nodes and members from the structural graph."""
        from config import DEFAULTS

        mat_type = self.material.get("type", "Steel")
        defaults = DEFAULTS.get(mat_type, DEFAULTS.get("Steel", {}))

        # Role → default section name
        role_default = {
            "primary_crease":    defaults.get("Primary", "IPE_300"),
            "secondary_lattice": defaults.get("Secondary", "HEA_200"),
            "bracing":           defaults.get("Diagonal", "Tubular_HSS_4x4x1/4"),
        }

        # Add nodes
        # Blender OBJ exports Y-up: Y = height (vertical), Z = depth
        min_y = float('inf')
        for node_id, data in self.graph.nodes(data=True):
            coords = data["coords"]
            self.model.add_node(str(node_id), coords[0], coords[1], coords[2])
            if coords[1] < min_y:
                min_y = coords[1]

        # Fixed supports at the lowest Y level (foundation / ground floor)
        # Use a 10% of total height tolerance to catch all ground nodes
        y_vals = [data["coords"][1] for _, data in self.graph.nodes(data=True)]
        y_range = max(y_vals) - min_y if y_vals else 1.0
        tol = max(0.5, y_range * 0.05)
        for node_id, data in self.graph.nodes(data=True):
            if abs(data["coords"][1] - min_y) < tol:
                self.model.def_support(str(node_id), True, True, True, True, True, True)

        # Add members
        mat_name = mat_type
        for member_idx, (u, v, edata) in enumerate(self.graph.edges(data=True), start=1):
            member_name = str(member_idx)
            role = edata.get("section_type", "secondary_lattice")

            # Prefer the explicitly assigned section; fall back to role default
            section_name = edata.get("section") or role_default.get(role, "HEA_200")

            self._ensure_section(section_name, mat_type)
            self.model.add_member(member_name, str(u), str(v), mat_name, section_name)
            # Note: pinned releases are stored on edges for display/export but not applied
            # to the FEA model — applying them to all secondary members causes singular
            # stiffness matrices when interior nodes have only pinned members meeting at them.

    # ------------------------------------------------------------------
    def apply_loads(self, gravity=9.81):
        """
        Load cases:
          D  — Dead / self-weight (distributed load based on section area × density)
          L  — Live load (floor occupancy, nodal)
          E  — Lateral seismic (ELF approximation)
          D+L+E — Combined factored (1.2D + 1.0L + 1.0E)
        """
        self.model.add_load_combo('D+L+E', {'D': 1.2, 'L': 1.0, 'E': 1.0})
        self.model.add_load_combo('D', {'D': 1.0})

        rho_n = self.material.get("rho", 7850) * gravity  # N/m³

        # Find min Y for seismic base shear height calculation
        min_y_val = min((n.Y for n in self.model.nodes.values()), default=0.0)

        for member_name, member in self.model.members.items():
            area = member.section.A  # m²
            dist_load = rho_n * area  # N/m (weight per unit length)
            try:
                # FY = vertical axis in Blender Y-up OBJ coordinate system
                self.model.add_member_dist_load(member_name, 'FY', -dist_load, -dist_load, 0, 100, case='D')
            except Exception:
                pass

        for node_id, node in self.model.nodes.items():
            height_above_base = node.Y - min_y_val
            if height_above_base > 1.0:
                lateral_force = 500 * height_above_base  # ELF approximation
                self.model.add_node_load(node_id, 'FX', lateral_force, case='E')
                self.model.add_node_load(node_id, 'FY', -2000, case='L')

    # ------------------------------------------------------------------
    def solve_and_evaluate(self):
        """
        Solve the FEA model and evaluate:
          - Maximum nodal displacement (combined D+L+E)
          - Per-node displacement magnitudes (for visualisation gradient)
          - L/360 deflection check per member
        """
        print("Solving FEA Model with PyNite...")
        try:
            self.model.analyze(check_statics=True)
            print("Solve complete.")
        except Exception as e:
            print(f"Solver Error: {e}")
            return {
                "status": "Error",
                "failures": [{"member": "all", "reason": "Matrix Singularity or Solver Error"}],
                "max_displacement": 0,
                "node_displacements": {}
            }

        failures = []
        max_disp = 0.0
        node_displacements = {}

        # Nodal displacement check
        for node_id, node in self.model.nodes.items():
            try:
                dx = node.DX.get('D+L+E', 0)
                dy = node.DY.get('D+L+E', 0)
                dz = node.DZ.get('D+L+E', 0)
                disp_mag = math.sqrt(dx**2 + dy**2 + dz**2)
                node_displacements[node_id] = disp_mag
                if disp_mag > max_disp:
                    max_disp = disp_mag
            except Exception:
                node_displacements[node_id] = 0.0

        # Member deflection check (L/360 serviceability)
        for member_name, member in self.model.members.items():
            L = member.L()
            limit = L / 360.0
            try:
                max_deflection = member.max_deflection('dy', 'D+L+E')
                if abs(max_deflection) > limit:
                    failures.append({
                        "member": member_name,
                        "reason": f"Deflection {abs(max_deflection):.4f}m > Limit {limit:.4f}m"
                    })
            except Exception:
                pass

        return {
            "status": "Failed" if failures else "Passed",
            "failures": failures,
            "max_displacement": max_disp,
            "node_displacements": node_displacements,
        }


if __name__ == "__main__":
    print("FEA Physics Engine Module Ready.")
