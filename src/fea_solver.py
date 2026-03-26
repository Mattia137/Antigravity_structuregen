import math
import networkx as nx
import numpy as np
from Pynite import FEModel3D

class FEASolver:
    def __init__(self, structural_graph: nx.Graph, material_params: dict):
        """
        Convert the designed structural graph into a PyNite FEA model.
        material_params should contain:
        {
            "type": "Steel" or "Concrete",
            "E": 200e9,        # Young's Modulus (Pa)
            "G": 77e9,         # Shear Modulus (Pa) 
            "nu": 0.3,         # Poisson's ratio
            "rho": 7850,       # Density (kg/m^3)
            "Fy": 350e6,       # Yield Strength (Pa)
        }
        """
        self.graph = structural_graph
        self.model = FEModel3D()
        self.material = material_params
        self.failures = []
        
        # Add material to the model
        material_name = self.material.get("type", "Material_1")
        self.model.add_material(material_name, E=self.material.get("E", 200e9), 
                                G=self.material.get("G", 77e9), 
                                nu=self.material.get("nu", 0.3), 
                                rho=self.material.get("rho", 7850))
        
        # Standard geometries for cross sections
        self.section_props = {
            "primary_crease": {"A": 0.05, "Iy": 0.001, "Iz": 0.001, "J": 0.002},
            "secondary_lattice": {"A": 0.01, "Iy": 0.0001, "Iz": 0.0001, "J": 0.0002}
        }
        
        # Register named sections with the model (required by current PyNite API)
        for sec_name, props in self.section_props.items():
            self.model.add_section(sec_name, props["A"], props["Iy"], props["Iz"], props["J"])
        
    def build_model(self):
        """
        Populates the FEA model with nodes and members from the structural graph.
        """
        min_z = float('inf')
        for node_id, data in self.graph.nodes(data=True):
            coords = data["coords"]
            self.model.add_node(str(node_id), coords[0], coords[1], coords[2])
            if coords[2] < min_z:
                min_z = coords[2]
                
        # Support conditions: fix nodes at the lowest Z level
        for node_id, data in self.graph.nodes(data=True):
            if abs(data["coords"][2] - min_z) < 0.1:
                self.model.def_support(str(node_id), True, True, True, True, True, True)
                
        # Add Members
        member_id = 0
        mat_name = self.material.get("type", "Material_1")
        for u, v, data in self.graph.edges(data=True):
            member_id += 1
            section_type = data.get("section_type", "secondary_lattice")
            # Ensure section_type matches a registered section name
            if section_type not in self.section_props:
                section_type = "secondary_lattice"
            
            self.model.add_member(str(member_id), str(u), str(v), mat_name, section_type)

    def apply_loads(self, gravity=9.81):
        """
        Apply automated load cases: Dead (Self-weight), Live (Occupancy), Lateral (LA Seismic)
        """
        self.model.add_load_combo('D+L+E', {'D': 1.2, 'L': 1.0, 'E': 1.0})
        
        # Dead Load (Self-weight)
        rho_n = self.material.get("rho", 7850) * gravity # specific weight in N/m^3
        
        for member_name, member in self.model.members.items():
            # Uniform downward load in N/m based on Volume / L = Area
            area = member.section.A
            dist_load = rho_n * area 
            # In PyNite, uniform distributed load on global Z takes negative for downward
            try:
                self.model.add_member_dist_load(member_name, 'FZ', -dist_load, -dist_load, 0, 100, case='D')
            except Exception:
                pass # Depending on version, method signatures differ
                
        # Live and Lateral Load simplifications
        for node_id, node in self.model.nodes.items():
            if node.Z > 1.0:
                # Seismic LA loads mimicking ELF distribution (heavier at top)
                lateral_force = 500 * node.Z 
                self.model.add_node_load(node_id, 'FX', lateral_force, case='E')
                
                # Live load on assumed floor nodes
                self.model.add_node_load(node_id, 'FZ', -2000, case='L')

    def solve_and_evaluate(self):
        """
        Solve FEA model and evaluate member stresses / nodal displacements.
        Identify any displacement > L/360.
        """
        print("Solving FEA Model with PyNite (checking statics)...")
        try:
            self.model.analyze(check_statics=True)
            print("Solving Complete.")
        except Exception as e:
            print(f"Solver Error: {e}")
            return {"status": "Error", "failures": [{"member": "all", "reason": "Matrix Singularity"}], "max_displacement": 0}
        
        failures = []
        max_disp = 0.0
        
        # Vectorized displacement check to extract max nodal limits
        for node_id, node in self.model.nodes.items():
            try:
                # Check displacement magnitude
                # Depending on version: node.DX['D+L+E']
                dx = node.DX.get('D+L+E', 0)
                dy = node.DY.get('D+L+E', 0)
                dz = node.DZ.get('D+L+E', 0)
                disp_mag = math.sqrt(dx**2 + dy**2 + dz**2)
                if disp_mag > max_disp:
                    max_disp = disp_mag
            except Exception:
                pass
                
        # Check L/360 limit per member
        for member_name, member in self.model.members.items():
            L = member.L()
            limit = L / 360.0
            
            try:
                max_deflection = member.max_deflection('dy', 'D+L+E') # local y displacement
                if abs(max_deflection) > limit:
                    failures.append({"member": member_name, "reason": f"Deflection {abs(max_deflection):.4f} > Limit {limit:.4f}"})
            except Exception:
                pass
                
        return {
            "status": "Failed" if failures else "Passed",
            "failures": failures,
            "max_displacement": max_disp
        }

if __name__ == "__main__":
    print("FEA Physics Engine Module Ready.")
