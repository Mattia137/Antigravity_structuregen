import os
import json
import networkx as nx
import google.generativeai as genai

class AIDesigner:
    def __init__(self, manual_path="knowledge_base/structural_manual.md"):
        """
        Connect to the external Gemini API using the GEMINI_AGENT_01 environment variable.
        """
        api_key = os.environ.get("GEMINI_AGENT_01", "DUMMY_KEY_FOR_TESTING")
        genai.configure(api_key=api_key)
        # Using the ultra-fast Flash model to minimize architecture latency while maintaining JSON schema precision
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                self.structural_manual = f.read()
        except FileNotFoundError:
            self.structural_manual = "Structural engineering manual not found. Defaulting to standard physics."

    def request_design(self, geometry_data: dict) -> dict:
        """
        Package the raw vertices, creases, SqFt data, and a summary of structural_manual.md into a system prompt.
        Commands the API to Design and return a structured JSON.
        """
        system_prompt = f"""
        You are the Lead Computational Structural Engineer.
        Knowledge base constraining your design logic:
        {self.structural_manual}
        
        Mandatory Generative Design Tasks:
        1. Receive the extracted vertices, creases (>20 deg dihedral), and SqFt metrics.
        2. Assign heavy cross-sections ('primary_crease') to the primary crease lines.
        3. Generate a secondary triangulated lattice connecting the remaining boundary nodes to prevent local buckling.
        4. Define exact X,Y coordinates for LABC-compliant concrete shear cores (considering aspect ratios and SqFt).
        5. Return exclusively a structured JSON defining the adjacency matrix (connectivity between nodes), cross-sections, and core placement.
        
        JSON Schema requirement:
        - "nodes": list of {{"id": int, "x": float, "y": float, "z": float}}
        - "edges": list of {{"source": int, "target": int, "type": string (either 'primary_crease' or 'secondary_lattice')}}
        - "cores": list of {{"x": float, "y": float, "thickness": float}}
        """
        
        # In a real environment, large geometry_data JSON needs to be within token limits.
        user_message = f"Extracted Geometry Data Payload:\n{json.dumps(geometry_data)}"
        
        try:
            # We enforce JSON response
            response = self.model.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_message}]}
                ],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            return json.loads(response.text)
        except Exception as e:
            import traceback
            with open("ai_crash.log", "w") as f:
                f.write(traceback.format_exc())
            print(f"Gemini API failed: {e}. Using geometric fallback.")
            return self._geometric_fallback(geometry_data)
    
    def _geometric_fallback(self, geometry_data: dict) -> dict:
        """
        When the AI API is unavailable, generate a valid structural frame
        directly from the input geometry bounding box.
        """
        verts = geometry_data.get("vertices", [])
        if not verts:
            verts = [[0,0,0], [10,0,0], [10,10,0], [0,10,0],
                     [0,0,10], [10,0,10], [10,10,10], [0,10,10]]
        
        import numpy as _np
        arr = _np.array(verts)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        cx, cy = (mins[0]+maxs[0])/2, (mins[1]+maxs[1])/2
        
        x0, x1 = float(mins[0]), float(maxs[0])
        y0, y1 = float(mins[1]), float(maxs[1])
        z0, z1 = float(mins[2]), float(maxs[2])
        
        height = z1 - z0
        floor_h = max(3.0, height / max(1, int(height / 3.0)))
        n_floors = max(1, int(height / floor_h))
        
        nodes = []
        edges = []
        nid = 0
        
        # Create a column-grid frame: 4 corners + center at each floor level
        floor_nodes = {}
        for fi in range(n_floors + 1):
            z = float(z0 + fi * floor_h)
            corners = [
                (x0, y0, z), (x1, y0, z),
                (x1, y1, z), (x0, y1, z),
                (float(cx), float(cy), z)
            ]
            level_ids = []
            for (px, py, pz) in corners:
                nodes.append({"id": nid, "x": px, "y": py, "z": pz})
                level_ids.append(nid)
                nid += 1
            floor_nodes[fi] = level_ids
            
            # Horizontal ring beams at this floor
            for i in range(4):
                edges.append({"source": level_ids[i], "target": level_ids[(i+1)%4], "type": "primary_crease"})
            # Diagonals to center
            for i in range(4):
                edges.append({"source": level_ids[i], "target": level_ids[4], "type": "secondary_lattice"})
        
        # Vertical columns connecting floors
        for fi in range(n_floors):
            for ci in range(5):
                edges.append({"source": floor_nodes[fi][ci], "target": floor_nodes[fi+1][ci], "type": "primary_crease"})
            # X-bracing on each face
            for ci in range(4):
                edges.append({"source": floor_nodes[fi][ci], "target": floor_nodes[fi+1][(ci+1)%4], "type": "secondary_lattice"})
        
        cores = [{"x": float(cx), "y": float(cy), "thickness": 0.3}]
        
        print(f"Geometric fallback generated: {len(nodes)} nodes, {len(edges)} edges, {n_floors} floors")
        return {"nodes": nodes, "edges": edges, "cores": cores}

    def construct_graph(self, design_json: dict) -> nx.Graph:
        """
        Construct the 3D structural graph in Python using the API's response.
        Utilizes networkx to build the adjacency and node attributes.
        """
        G = nx.Graph()
        
        # Add nodes
        for node in design_json.get("nodes", []):
            G.add_node(node["id"], coords=(node["x"], node["y"], node["z"]))
            
        # Add edges (adjacency matrix)
        for edge in design_json.get("edges", []):
            G.add_edge(edge["source"], edge["target"], section_type=edge.get("type", "secondary_lattice"))
            
        # Store LABC shear core locations as graph-level metadata
        G.graph["shear_cores"] = design_json.get("cores", [])
        
        print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

if __name__ == "__main__":
    designer = AIDesigner()
    print("AI Generative Designer Ready.")
