import os
import json
import networkx as nx
from google import genai
from google.genai import types


class AIDesigner:
    def __init__(
        self,
        manual_path="knowledge_base/structural_manual.md",
        research_path="research/structural_patterns.md",
    ):
        """
        Connect to the Gemini API.
        Key lookup: GEMINI_AGENT_02 → GEMINI_AGENT_01 → GOOGLE_API_KEY
        """
        self.api_key = (
            os.getenv("GEMINI_AGENT_02")
            or os.getenv("GEMINI_AGENT_01")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_AGENT_02, GEMINI_AGENT_01, "
                "or GOOGLE_API_KEY."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-1.5-flash"

        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                self.structural_manual = f.read()
        except FileNotFoundError:
            self.structural_manual = "Structural engineering manual not found."

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def request_base_design(self, geometry_data: dict) -> dict:
        """
        Phase 2: Ask Gemini to output ONLY structural topology (nodes + edges).
        Section assignment is handled deterministically by section_sizer.py.

        Returns the parsed JSON dict or calls _geometric_fallback on failure.
        """
        primary_nodes  = geometry_data.get("primary_nodes", [])
        primary_edges  = geometry_data.get("primary_edges", [])
        story_levels   = geometry_data.get("story_levels", [])
        peak_points    = geometry_data.get("peak_points", [])
        bounds         = geometry_data.get("bounds", {})
        rules_digest   = geometry_data.get("rules_digest", "")
        feedback       = geometry_data.get("optimization_feedback", "")

        system_prompt = f"""You are a Lead Computational Structural Engineer.

Your ONLY task: decide WHERE to place structural members on this mesh.
Cross-section sizing is handled separately — DO NOT assign sections.

{rules_digest}

STRUCTURAL KNOWLEDGE:
{self.structural_manual}

═══ RULES FOR TOPOLOGY OUTPUT ═══

STEP 1 — PRESERVE PRIMARY NODES (MANDATORY):
  Output ALL provided primary_nodes with their EXACT (x, y, z) coordinates.
  Do NOT modify or drop any node that was given to you.

STEP 2 — USE PRIMARY EDGES AS STRUCTURAL BACKBONE:
  Output ALL primary_edges as the structural spine.

STEP 3 — ADD SECONDARY NODES where structurally required:
  · Mid-height bracing when a vertical run exceeds 6 m.
  · Floor-level ring nodes to close horizontal diaphragms.
  · Intersection nodes where diagonal braces would cross.
  Secondary node IDs start at max(primary_node_id) + 1.
  Add only what is structurally necessary.

STEP 4 — ADD SECONDARY EDGES:
  Connect secondary nodes with bracing and ring beams.
  Use type "secondary_lattice" for all added edges.

STEP 5 — CORES:
  Place at least 1 concrete shear core at the building centroid.
  More cores if the plan is wide (>30 m).

{f"REVISION REQUIRED — fix these issues: {feedback}" if feedback else ""}

OUTPUT: Return ONLY valid JSON (no markdown, no explanation):
{{
  "nodes": [{{"id": int, "x": float, "y": float, "z": float}}],
  "edges": [{{"source": int, "target": int, "type": "primary_crease|secondary_lattice"}}],
  "cores": [{{"x": float, "y": float, "thickness": float, "width": float, "depth": float}}]
}}
"""

        user_message = (
            f"STRUCTURAL NODES ({len(primary_nodes)} nodes):\n"
            f"{json.dumps(primary_nodes)}\n\n"
            f"STRUCTURAL EDGES ({len(primary_edges)} edges):\n"
            f"{json.dumps(primary_edges)}\n\n"
            f"STORY LEVELS (Z coordinates): {json.dumps(story_levels)}\n"
            f"PEAK POINTS (roof centroid candidates): {json.dumps(peak_points)}\n"
            f"BOUNDING BOX (metres): {json.dumps(bounds)}"
        )

        print(f"Requesting AI topology ({len(primary_nodes)} nodes, {len(primary_edges)} edges)...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    max_output_tokens=2048,
                    temperature=0.2,
                ),
            )

            raw = response.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```json"):
                raw = raw[7:].rstrip("`").strip()
            elif raw.startswith("```"):
                raw = raw[3:].rstrip("`").strip()

            result = json.loads(raw)

            # Validate: primary nodes must all be present
            if not result.get("nodes") or len(result["nodes"]) < len(primary_nodes):
                print("Gemini dropped primary nodes — using geometric fallback.")
                return self._geometric_fallback(geometry_data)

            return result

        except Exception as e:
            import traceback
            with open("ai_crash.log", "a") as f:
                f.write(f"\nTopology generation error:\n")
                f.write(traceback.format_exc())
                if "response" in dir():
                    try:
                        f.write(f"Raw response: {response.text}\n")
                    except Exception:
                        pass
            print(f"Gemini API failed: {e} — using geometric fallback.")
            return self._geometric_fallback(geometry_data)

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback (no API)
    # ──────────────────────────────────────────────────────────────────────────

    def _geometric_fallback(self, geometry_data: dict) -> dict:
        """
        Build a minimal structural topology from the primary crease geometry.
        No section assignment here — section_sizer handles that.
        """
        import numpy as _np

        primary_nodes = geometry_data.get("primary_nodes", [])
        primary_edges = geometry_data.get("primary_edges", [])

        if not primary_nodes:
            return self._generic_cube_fallback()

        nodes = [
            {"id": n["id"], "x": n["x"], "y": n["y"], "z": n["z"]}
            for n in primary_nodes
        ]

        edges = [
            {"source": e["source"], "target": e["target"], "type": e.get("type", "primary_crease")}
            for e in primary_edges
        ]

        # Add centroid stabilisers at each Z-level if sparse
        arr = _np.array([[n["x"], n["y"], n["z"]] for n in nodes])
        if len(nodes) < 50:
            nid = max(n["id"] for n in nodes) + 1
            z_vals = _np.round(arr[:, 2], 1)
            for z_level in _np.unique(z_vals):
                mask = _np.abs(arr[:, 2] - z_level) < 0.5
                ids_at_level = [nodes[i]["id"] for i in _np.where(mask)[0]]
                if len(ids_at_level) < 3:
                    continue
                cx = float(arr[mask, 0].mean())
                cy = float(arr[mask, 1].mean())
                nodes.append({"id": nid, "x": cx, "y": cy, "z": float(z_level)})
                for fid in ids_at_level:
                    edges.append({"source": fid, "target": nid, "type": "secondary_lattice"})
                nid += 1

        peak_points = geometry_data.get("peak_points", [[0, 0]])
        cores = [
            {"x": p[0], "y": p[1], "thickness": 0.3, "width": 6.0, "depth": 6.0}
            for p in peak_points
        ]

        return {"nodes": nodes, "edges": edges, "cores": cores}

    def _generic_cube_fallback(self) -> dict:
        """Ultra-fallback — simple 3D frame when no geometry is available."""
        nodes = [
            {"id": 0, "x": 0,  "y": 0,  "z": 0 }, {"id": 1, "x": 10, "y": 0,  "z": 0 },
            {"id": 2, "x": 10, "y": 10, "z": 0 }, {"id": 3, "x": 0,  "y": 10, "z": 0 },
            {"id": 4, "x": 0,  "y": 0,  "z": 10}, {"id": 5, "x": 10, "y": 0,  "z": 10},
            {"id": 6, "x": 10, "y": 10, "z": 10}, {"id": 7, "x": 0,  "y": 10, "z": 10},
        ]
        edges = [
            {"source": 0, "target": 4, "type": "primary_crease"},
            {"source": 1, "target": 5, "type": "primary_crease"},
            {"source": 2, "target": 6, "type": "primary_crease"},
            {"source": 3, "target": 7, "type": "primary_crease"},
            {"source": 4, "target": 5, "type": "primary_crease"},
            {"source": 5, "target": 6, "type": "primary_crease"},
            {"source": 6, "target": 7, "type": "primary_crease"},
            {"source": 7, "target": 4, "type": "primary_crease"},
            {"source": 4, "target": 6, "type": "secondary_lattice"},
            {"source": 5, "target": 7, "type": "secondary_lattice"},
        ]
        cores = [{"x": 5.0, "y": 5.0, "thickness": 0.3, "width": 6.0, "depth": 6.0}]
        return {"nodes": nodes, "edges": edges, "cores": cores}

    # ──────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────────

    def construct_graph(self, design_json: dict) -> nx.Graph:
        """
        Build a NetworkX graph from the AI topology response.
        Section names are NOT set here — apply_sections() does that.
        """
        G = nx.Graph()

        for node in design_json.get("nodes", []):
            G.add_node(
                node["id"],
                coords=(node["x"], node["y"], node["z"]),
                connection_type=node.get("connection_type", "welded"),
            )

        for e in design_json.get("edges", []):
            G.add_edge(
                e["source"], e["target"],
                section_type=e.get("type", "secondary_lattice"),
                section=None,          # will be filled by section_sizer
                connection=None,       # will be filled by section_sizer
                typology=None,         # will be filled by section_sizer
            )

        G.graph["shear_cores"] = design_json.get("cores", [])
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G


if __name__ == "__main__":
    designer = AIDesigner()
    print("AI Topology Designer ready.")
