from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Importing the AI & Physical phases we just built
from src.geometry_engine import GeometryEngine
from src.ai_designer import AIDesigner
from src.optimizer import EvolutionaryOptimizer

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/mesh/<path:filename>')
def serve_mesh(filename):
    return send_from_directory('.', filename)

@app.route('/api/config')
def get_config():
    return jsonify({
        'materials': {
            'Steel': {'E': 29000, 'nu': 0.3, 'rho': 0.283, 'Strength': 50},
            'Concrete': {'E': 4000, 'nu': 0.2, 'rho': 0.086, 'Strength': 5}
        }
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    Hooks the web frontend directly into the AI Evolutionary Pipeline.
    """
    data = request.json
    mat_type = data.get('material', 'Steel')
    
    material_params = {
        "type": mat_type,
        "E": 200e9 if mat_type == 'Steel' else 30e9,
        "nu": 0.3 if mat_type == 'Steel' else 0.2,
        "rho": 7850.0 if mat_type == 'Steel' else 2400.0,
        "G": 77e9 if mat_type == 'Steel' else 12e9,
        "Fy": 350e6 if mat_type == 'Steel' else 40e6
    }

    # STEP 1: Geometry Extraction (Phase 2)
    # Using 'mass-DEF.obj' as default payload mock unless file handling is provided
    try:
        ge = GeometryEngine('mass-DEF.obj')
        base_geom = {
            "vertices": ge.extract_boundary_nodes(),
            "creases": ge.extract_primary_creases(),
            "sqft_data": ge.slice_mesh_horizontally()
        }
    except Exception as e:
        print(f"Geometry Extraction error: {e}")
        base_geom = {"vertices": [[0,0,0], [10,0,0], [10,10,0], [0,10,0], [0,0,10], [10,0,10], [10,10,10], [0,10,10]]}

    # STEP 2: Generative Optimizer (Phase 3 & 5)
    ai = AIDesigner()
    opt = EvolutionaryOptimizer(ai)
    
    # We run 1 max iteration to prevent Hugging Face's 60-second proxy timeout latency
    final_graph, best_results = opt.run_optimization_loop(base_geom, material_params, max_iterations=1)
    
    if not final_graph or final_graph.number_of_nodes() == 0:
        return jsonify({'error': 'AI generative loop failed to produce graph nodes.'}), 500
        
    max_disp = best_results.get("max_displacement", 0.001) if best_results else 0.01

    # STEP 3: Mapping back to Frontend format
    nodes_out = {}
    for node_id, ndata in final_graph.nodes(data=True):
        coords = ndata["coords"]
        nodes_out[str(node_id)] = {
            "x": coords[0],
            "y": coords[1],
            "z": coords[2]
        }
        
    members_out = []
    total_length = 0.0
    edge_idx = 0
    
    for u, v, edata in final_graph.edges(data=True):
        edge_idx += 1
        
        # Calculate displacement gradient color proxy loosely based on height or FEA fallback
        p1 = np.array(final_graph.nodes[u]["coords"])
        p2 = np.array(final_graph.nodes[v]["coords"])
        total_length += np.linalg.norm(p1 - p2)
        
        # Proxy displacement (usually extracted directly from PyNite via node.DX)
        disp_i_perc = abs(p1[2] / 20.0) # height based displacement
        disp_j_perc = abs(p2[2] / 20.0)
        
        members_out.append({
            "id": f"m_{edge_idx}",
            "from": str(u),
            "to": str(v),
            "role": edata.get("section_type", "secondary_lattice"),
            "section": mat_type,
            "disp_i": disp_i_perc,
            "disp_j": disp_j_perc
        })

    # Sustainability and Carbon calculation
    total_volume = total_length * 0.05
    total_mass = total_volume * material_params["rho"]
    
    if mat_type == "Steel":
        total_carbon = total_mass * 1.22
        total_cost = (total_mass / 1000) * 2653.0
    else:
        total_carbon = total_mass * 0.20
        total_cost = (total_mass / 1833) * 145.0

    return jsonify({
        'metrics': {
            'Carbon_kgCO2e': round(total_carbon, 0),
            'Cost_USD': round(total_cost, 0),
            'Volume': round(total_volume, 2),
            'Max_Disp': round(max_disp, 4)
        },
        'nodes': nodes_out,
        'members': members_out,
        'max_disp': max_disp,
        'unit': 'm'
    })

if __name__ == '__main__':
    print("  AI GENERATIVE SERVER running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
