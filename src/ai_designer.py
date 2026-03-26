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
    def __init__(self, manual_path="knowledge_base/structural_manual.md"):
        """
        Connect to the external Gemini API using the GEMINI_AGENT_01 environment variable.
        """
        api_key = os.environ.get("GEMINI_AGENT_01", "DUMMY_KEY_FOR_TESTING")
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-1.5-flash'

        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                self.structural_manual = f.read()
        except FileNotFoundError:
            self.structural_manual = "Structural engineering manual not found. Defaulting to standard physics."

    def request_design(self, geometry_data: dict) -> dict:
        """
        Build a Gemini prompt that:
          1. Takes mesh crease vertices as PRIMARY nodes (coordinates must be preserved exactly).
          2. Takes mesh crease edges as PRIMARY structural members.
          3. Asks Gemini to add SECONDARY nodes/edges only where structurally necessary.
          4. Assigns a cross-section and connection type to every edge.
        """
        primary_nodes = geometry_data.get("primary_nodes", [])
        primary_edges = geometry_data.get("primary_edges", [])
        sqft_data = geometry_data.get("sqft_data", {})
        bounds = geometry_data.get("bounds", {})
        feedback = geometry_data.get("optimization_feedback", "")

        # Infer material from feedback context (default Steel)
        section_list = STEEL_SECTIONS

        system_prompt = f"""You are the Lead Computational Structural Engineer performing generative structural design.

KNOWLEDGE BASE:
{self.structural_manual}

---
WORKFLOW:

You receive:
- "primary_nodes": nodes extracted from mesh crease vertices — these are the PRIMARY structural nodes.
  Their (x, y, z) coordinates are EXACT and must NOT be modified.
- "primary_edges": edges extracted from mesh crease lines — these are the PRIMARY structural members.
- "sqft_data": floor area data.
- "bounds": overall mesh extents (meters).
- "optimization_feedback": FEA results from the previous iteration (empty on first run).

YOUR TASKS:

TASK 1 — PRESERVE PRIMARY STRUCTURE (mandatory):
  • Output ALL primary_nodes with their EXACT coordinates (copy id, x, y, z verbatim).
  • Output ALL primary_edges as type "primary_crease".

TASK 2 — ADD SECONDARY STRUCTURE (add only what is structurally necessary):
  • Evaluate the primary topology for:
      - Unbraced column lengths > L/360 serviceability limit
      - Missing lateral / X-bracing between floors
      - Out-of-plane instability zones
      - Local buckling risks in long primary members
  • If needed, add new secondary nodes (IDs starting after the last primary node ID)
    and secondary_lattice edges to address these issues.
  • Minimise secondary additions — only add what is required.

TASK 3 — ASSIGN CROSS-SECTIONS (every edge must have a "section"):
  Choose from this list only: {section_list}

  Assignment rules (IBC 2024 / AISC 360-22):
  - Vertical/near-vertical primary columns (tall or high axial): W14x283 or W14x159
  - Medium columns or raking members: W14x90 or W18x97
  - Long horizontal beams (span > 9m): W24x146 or W18x97
  - Medium beams (span 5-9m): W12x53 or W12x50
  - Short beams / ties (span < 5m): W8x31 or IPE_300
  - Large perimeter diagonal braces: HSS16x0.625 or HSS12x12x0.625
  - Medium diagonal braces: HSS10x0.500 or HSS8x8x0.500
  - Light secondary lattice / bracing: HEA_200, HSS6x0.500, or Tubular_HSS_4x4x1/4
  - Assign heavier sections at high-connectivity nodes (degree >= 4) and long spans.

TASK 4 — ASSIGN CONNECTION TYPES (every edge must have a "connection"):
  - "fixed": moment-resisting connection — use for beams, continuous columns, moment frames
  - "pinned": pin/truss connection — use for bracing members, secondary lattice, short ties

TASK 5 — SHEAR CORE PLACEMENT:
  Place LABC-compliant concrete shear cores per the knowledge base rules (centralized, C-shape around service cores).

{f"OPTIMIZATION FEEDBACK FROM PREVIOUS FEA RUN: {feedback}" if feedback else ""}

OUTPUT: Return ONLY this JSON (no markdown, no extra text):
{{
  "nodes": [{{"id": int, "x": float, "y": float, "z": float}}],
  "edges": [{{"source": int, "target": int, "type": "primary_crease|secondary_lattice", "section": "string", "connection": "fixed|pinned"}}],
  "cores": [{{"x": float, "y": float, "thickness": float}}]
}}
"""

        user_message = (
            f"PRIMARY NODES ({len(primary_nodes)} nodes):\n{json.dumps(primary_nodes)}\n\n"
            f"PRIMARY EDGES ({len(primary_edges)} edges):\n{json.dumps(primary_edges)}\n\n"
            f"SQFT DATA: {json.dumps(sqft_data)}\n"
            f"BOUNDS: {json.dumps(bounds)}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt + "\n\n" + user_message,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            # Validate that primary nodes are present in the output
            if not result.get("nodes") or len(result["nodes"]) < len(primary_nodes):
                print("Gemini response missing primary nodes. Falling back.")
                return self._geometric_fallback(geometry_data)
            return result
        except Exception as e:
            import traceback
            with open("ai_crash.log", "w") as f:
                f.write(traceback.format_exc())
            print(f"Gemini API failed: {e}. Using geometric fallback.")
            return self._geometric_fallback(geometry_data)

    def _geometric_fallback(self, geometry_data: dict) -> dict:
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

        # Assign sections based on edge type
        edges = []
        for e in primary_edges:
            etype = e.get("type", "secondary_lattice")
            section = "IPE_300" if etype == "primary_crease" else "HEA_200"
            edges.append({
                "source": e["source"], "target": e["target"],
                "type": etype, "section": section, "connection": "fixed"
            })

        arr = _np.array([[n["x"], n["y"], n["z"]] for n in nodes])

        # Only add centroid stabilisers when the primary structure is sparse
        if len(nodes) < 20:
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
                        "type": "secondary_lattice", "section": "HEA_200", "connection": "fixed"
                    })
                nid += 1

        cores = [{"x": float(arr[:, 0].mean()), "y": float(arr[:, 1].mean()), "thickness": 0.3}]
        print(f"Fallback: {len(nodes)} nodes, {len(edges)} edges")
        return {"nodes": nodes, "edges": edges, "cores": cores}

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

        for edge in design_json.get("edges", []):
            G.add_edge(
                edge["source"], edge["target"],
                section_type=edge.get("type", "secondary_lattice"),
                section=edge.get("section", None),
                connection=edge.get("connection", "fixed")
            )

        G.graph["shear_cores"] = design_json.get("cores", [])
        print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G


if __name__ == "__main__":
    designer = AIDesigner()
    print("AI Generative Designer Ready.")
