import os
import json
import networkx as nx
from google import genai
from google.genai import types

# Available cross-section names (must match config.py SECTIONS keys)
STEEL_SECTIONS = [
    "W8x31", "W12x50", "W12x53", "W14x90", "W14x159", "W14x283",
    "W18x97", "W24x146", "IPE_300", "HEA_200",
    "HSS6x0.500", "HSS10x0.500", "HSS16x0.625",
    "Tubular_HSS_4x4x1/4", "HSS8x8x0.500", "HSS12x12x0.625"
]
CONCRETE_SECTIONS = ["Rect_16x16", "Circ_16", "Floor_Tie", "Core_Massive"]

class AIDesigner:
    def __init__(self, manual_path="knowledge_base/structural_manual.md", research_path="research/structural_patterns.md"):
        """
        Connect to the external Gemini API using the GEMINI_AGENT_01 environment variable.
        """
        self.api_key = os.getenv("GEMINI_AGENT_02") or os.getenv("GEMINI_AGENT_01") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set one of the following environment variables: "
                "GEMINI_AGENT_02, GEMINI_AGENT_01, or GOOGLE_API_KEY."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = 'gemini-1.5-flash'

        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                self.structural_manual = f.read()
        except FileNotFoundError:
            self.structural_manual = "Structural engineering manual not found."

        try:
            with open(research_path, "r", encoding="utf-8") as f:
                self.research_patterns = f.read()
        except FileNotFoundError:
            self.research_patterns = "No research patterns found."

    def request_design(self, geometry_data: dict, optimization_goal: str = "BALANCED") -> dict:
        """
        Build a Gemini prompt that:
          1. Takes mesh crease vertices as PRIMARY nodes.
          2. Takes mesh crease edges as PRIMARY structural members.
          3. Uses INTERNAL_NODES as high-density anchor points for a complex lattice.
          4. Optimizes for the specified goal (COST, CARBON, or BALANCED).
        """
        primary_nodes = geometry_data.get("primary_nodes", [])
        primary_edges = geometry_data.get("primary_edges", [])
        internal_nodes = geometry_data.get("internal_nodes", [])
        sqft_data = geometry_data.get("sqft_data", {})
        bounds = geometry_data.get("bounds", {})
        feedback = geometry_data.get("optimization_feedback", "")

        section_list = STEEL_SECTIONS

        system_prompt = f"""You are the Lead Computational Structural Engineer performing generative structural design.
        
GOAL: Optimize for {optimization_goal}
- COST: Minimize total material weight and connection complexity.
- CARBON: Prioritize low-carbon sections and efficient geometries.
- BALANCED: Professional trade-off between displacement and footprint.

KNOWLEDGE BASE:
{self.structural_manual}

RESEARCH PATTERNS:
{self.research_patterns}

---
WORKFLOW:
You receive:
- "primary_nodes": coordinates of mesh vertices (EXACT).
- "primary_edges": mesh crease lines (PRIMARY frame).
- "internal_nodes": sampled 3D points INSIDE the volume for high-density lattice anchors.
- "sqft_data": floor area and centroids.
- "optimization_feedback": FEA results for iteration.

TASKS:
1. PRESERVE PRIMARY STRUCTURE: Keep all primary_nodes and primary_edges.
2. REARRANGE & EXPAND:
   - Rearrange non-crease edges to optimize load paths.
   - Add nodes from "internal_nodes" to create a complex internal structural network.
   - Triangulate for global stability (see "Triangulation" in research patterns).
3. ASSIGN SECTIONS: Choose from {section_list} based on manual rules.
4. IDENTIFY CONNECTION TYPOLOGY (MANDATORY):
   - `welded`: High-moment primary joints (beam-to-column, primary frame nodes).
   - `hinge`: Secondary diagonal braces and pinned lattice nodes.
   - `steel_plate`: Secondary beam-to-column or tie-beam connections.
5. PLACE CONCRETE CORES (CODE COMPLIANCE): 
   - Min 1.5% GFA. 
   - Position cores at "PEAK POINTS" (Max Height) to ensure elevators reach the rooftop.
   - Ensure the "SUGGESTED CORE COUNT" is met for large buildings (every floor point must be within 60m of a core).

OUTPUT: Return ONLY JSON:
{{
  "nodes": [{{"id": int, "x": float, "y": float, "z": float}}],
  "edges": [{{"source": int, "target": int, "type": "primary_crease|secondary_lattice", "section": "string", "connection": "fixed|pinned", "typology": "welded|hinge|steel_plate"}}],
  "cores": [{{"x": float, "y": float, "thickness": float, "width": float, "depth": float}}]
}}
"""

        peak_points = geometry_data.get("peak_points", [])

        user_message = (
            f"PRIMARY NODES ({len(primary_nodes)} nodes):\n{json.dumps(primary_nodes)}\n\n"
            f"PRIMARY EDGES ({len(primary_edges)} edges):\n{json.dumps(primary_edges)}\n\n"
            f"INTERNAL NODES ({len(internal_nodes)} sample points):\n{json.dumps(internal_nodes[:100])}\n\n"
            f"PEAK POINTS (Max Height locations): {json.dumps(peak_points)}\n"
            f"SQFT DATA & SUGGESTED CORES: {json.dumps(sqft_data)}\n"
            f"BOUNDS: {json.dumps(bounds)}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt + "\n\n" + user_message,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            
            # Ensure optimization_goal is tagged in the result for tracking
            result["optimization_goal"] = optimization_goal
            
            # Validate that primary nodes are present in the output
            if not result.get("nodes") or len(result["nodes"]) < len(primary_nodes):
                print(f"Gemini response missing primary nodes for {optimization_goal}. Falling back.")
                return self._geometric_fallback(geometry_data, optimization_goal)
            return result
        except Exception as e:
            import traceback
            with open("ai_crash.log", "a") as f:
                f.write(f"\nAPI Error during {optimization_goal} generation:\n")
                f.write(traceback.format_exc())
            print(f"Gemini API failed for {optimization_goal}: {e}. Using geometric fallback.")
            return self._geometric_fallback(geometry_data, optimization_goal)

    def request_variants(self, geometry_data: dict) -> list:
        """
        Request 3 distinct design variants from Gemini: COST, CARBON, BALANCED.
        """
        variants = []
        for goal in ["COST", "CARBON", "BALANCED"]:
            print(f"Requesting AI Variant: {goal}...")
            design = self.request_design(geometry_data, optimization_goal=goal)
            variants.append(design)
        return variants

    def _geometric_fallback(self, geometry_data: dict, optimization_goal: str = "BALANCED") -> dict:
        """
        Fallback when Gemini is unavailable.
        Assigns sections to every primary edge. Adds centroid stabilisers only
        when the primary structure is sparse (< 20 nodes).
        """
        import numpy as _np

        primary_nodes = geometry_data.get("primary_nodes", [])
        primary_edges = geometry_data.get("primary_edges", [])

        # Legacy format compatibility
        if not primary_nodes:
            vertices = geometry_data.get("vertices", [])
            primary_nodes = [
                {"id": i, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
                for i, v in enumerate(vertices)
            ]

        if len(primary_nodes) < 2:
            return self._generic_cube_fallback()

        nodes = [{"id": n["id"], "x": n["x"], "y": n["y"], "z": n["z"]} for n in primary_nodes]

        # Different default sections depending on the optimization goal
        if optimization_goal == "COST":
            crease_sec = "W12x50"
            lattice_sec = "Tubular_HSS_4x4x1/4"
        elif optimization_goal == "CARBON":
            crease_sec = "IPE_300"
            lattice_sec = "HEA_200"
        else: # BALANCED
            crease_sec = "W14x90"
            lattice_sec = "W8x31"

        # Assign sections based on edge type
        edges = []
        for e in primary_edges:
            etype = e.get("type", "secondary_lattice")
            section = crease_sec if etype == "primary_crease" else lattice_sec
            edges.append({
                "source": e["source"], "target": e["target"],
                "type": etype, "section": section, "connection": "fixed"
            })

        arr = _np.array([[n["x"], n["y"], n["z"]] for n in nodes])

        # Add centroid stabilisers if sparse
        if len(nodes) < 50:
            nid = max(n["id"] for n in nodes) + 1
            z_vals = _np.round(arr[:, 2], 1)
            for z_level in _np.unique(z_vals):
                floor_mask = _np.abs(arr[:, 2] - z_level) < 0.5
                floor_indices = _np.where(floor_mask)[0]
                floor_ids = [nodes[i]["id"] for i in floor_indices]
                if len(floor_ids) < 3:
                    continue
                cx = float(arr[floor_mask, 0].mean())
                cy = float(arr[floor_mask, 1].mean())
                nodes.append({"id": nid, "x": cx, "y": cy, "z": float(z_level)})
                for fid in floor_ids:
                    edges.append({
                        "source": fid, "target": nid,
                        "type": "secondary_lattice", "section": lattice_sec, "connection": "fixed"
                    })
                nid += 1

        cores = [{"x": p[0], "y": p[1], "thickness": 0.3, "width": 6.0, "depth": 6.0} for p in geometry_data.get("peak_points", [[0,0]])]
        
        # Ensure fallback edges have typologies
        final_edges = []
        for e in edges:
            e["typology"] = "welded" if e["type"] == "primary_crease" else "hinge"
            final_edges.append(e)

        return {"nodes": nodes, "edges": final_edges, "cores": cores, "optimization_goal": optimization_goal}

    def _generic_cube_fallback(self) -> dict:
        """Ultra-fallback: simple 3D frame when no geometry is available."""
        nodes = [
            {"id": 0, "x": 0, "y": 0, "z": 0}, {"id": 1, "x": 10, "y": 0, "z": 0},
            {"id": 2, "x": 10, "y": 10, "z": 0}, {"id": 3, "x": 0, "y": 10, "z": 0},
            {"id": 4, "x": 0, "y": 0, "z": 10}, {"id": 5, "x": 10, "y": 0, "z": 10},
            {"id": 6, "x": 10, "y": 10, "z": 10}, {"id": 7, "x": 0, "y": 10, "z": 10},
        ]
        edges = [
            {"source": 0, "target": 4, "type": "primary_crease", "section": "W14x90", "connection": "fixed"},
            {"source": 1, "target": 5, "type": "primary_crease", "section": "W14x90", "connection": "fixed"},
            {"source": 2, "target": 6, "type": "primary_crease", "section": "W14x90", "connection": "fixed"},
            {"source": 3, "target": 7, "type": "primary_crease", "section": "W14x90", "connection": "fixed"},
            {"source": 4, "target": 5, "type": "primary_crease", "section": "IPE_300", "connection": "fixed"},
            {"source": 5, "target": 6, "type": "primary_crease", "section": "IPE_300", "connection": "fixed"},
            {"source": 6, "target": 7, "type": "primary_crease", "section": "IPE_300", "connection": "fixed"},
            {"source": 7, "target": 4, "type": "primary_crease", "section": "IPE_300", "connection": "fixed"},
            {"source": 4, "target": 6, "type": "secondary_lattice", "section": "Tubular_HSS_4x4x1/4", "connection": "pinned"},
            {"source": 5, "target": 7, "type": "secondary_lattice", "section": "Tubular_HSS_4x4x1/4", "connection": "pinned"},
        ]
        cores = [{"x": 5.0, "y": 5.0, "thickness": 0.3}]
        return {"nodes": nodes, "edges": edges, "cores": cores}

    def construct_graph(self, design_json: dict) -> nx.Graph:
        """
        Construct the 3D structural graph from the AI/fallback response.
        Stores section name and connection type on each edge.
        """
        G = nx.Graph()

        for node in design_json.get("nodes", []):
            G.add_node(node["id"], coords=(node["x"], node["y"], node["z"]))

        G.add_edges_from([
            (e["source"], e["target"], {
                "section_type": e.get("type", "secondary_lattice"),
                "section": e.get("section"),
                "connection": e.get("connection", "fixed"),
                "typology": e.get("typology", "welded")
            }) for e in design_json.get("edges", [])
        ])

        G.graph["shear_cores"] = design_json.get("cores", [])
        print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G


if __name__ == "__main__":
    designer = AIDesigner()
    print("AI Generative Designer Ready.")
