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
            print(f"Failed to generate AI structural design: {e}")
            # Fallback mock schema for testing resilience
            return {"nodes": [], "edges": [], "cores": []}

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
