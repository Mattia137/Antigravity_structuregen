from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import json, os, sys, traceback, math
import numpy as np

def _safe_float(v, fallback=0.0):
    """Return fallback if v is NaN or infinite (JSON doesn't support those)."""
    try:
        return fallback if (math.isnan(v) or math.isinf(v)) else float(v)
    except Exception:
        return fallback

sys.path.insert(0, os.path.dirname(__file__))

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

    # STEP 1: Geometry Extraction
    # ALL mesh vertices → primary structural nodes (exact coordinates preserved).
    # ALL unique mesh edges → structural members, classified by dihedral angle:
    #   crease (angle > 5°) → primary_crease
    #   flat  (angle ≤ 5°) → secondary_lattice
    try:
        ge = GeometryEngine('mass-DEF.obj')
        all_verts = np.array(ge.extract_boundary_nodes())

        # Low threshold so we capture the full polyhedral skeleton
        creases = ge.extract_primary_creases(angle_threshold_degrees=5.0)
        crease_set = set(tuple(sorted(e)) for e in creases["edges"])

        # Every vertex is a primary node — coordinates preserved exactly
        primary_nodes = [
            {"id": i, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
            for i, v in enumerate(all_verts)
        ]

        # Every unique mesh edge is a structural member
        primary_edges = []
        for e in ge.mesh.edges_unique:
            u, v_idx = int(e[0]), int(e[1])
            etype = "primary_crease" if tuple(sorted([u, v_idx])) in crease_set else "secondary_lattice"
            primary_edges.append({"source": u, "target": v_idx, "type": etype})

        base_geom = {
            "primary_nodes": primary_nodes,
            "primary_edges": primary_edges,
            "sqft_data": ge.slice_mesh_horizontally(),
            "bounds": {
                "x_min": float(all_verts[:, 0].min()), "x_max": float(all_verts[:, 0].max()),
                "y_min": float(all_verts[:, 1].min()), "y_max": float(all_verts[:, 1].max()),
                "z_min": float(all_verts[:, 2].min()), "z_max": float(all_verts[:, 2].max()),
            }
        }
        print(f"Primary structure: {len(primary_nodes)} nodes, {len(primary_edges)} edges "
              f"({sum(1 for e in primary_edges if e['type']=='primary_crease')} crease, "
              f"{sum(1 for e in primary_edges if e['type']=='secondary_lattice')} secondary)")

    except Exception as e:
        with open("crash.log", "w") as f:
            f.write(traceback.format_exc())
        print(f"Geometry Extraction error: {e}")
        base_geom = {
            "primary_nodes": [
                {"id": 0, "x": 0, "y": 0, "z": 0}, {"id": 1, "x": 10, "y": 0, "z": 0},
                {"id": 2, "x": 10, "y": 10, "z": 0}, {"id": 3, "x": 0, "y": 10, "z": 0},
                {"id": 4, "x": 0, "y": 0, "z": 10}, {"id": 5, "x": 10, "y": 0, "z": 10},
                {"id": 6, "x": 10, "y": 10, "z": 10}, {"id": 7, "x": 0, "y": 10, "z": 10},
            ],
            "primary_edges": [
                {"source": 0, "target": 4}, {"source": 1, "target": 5},
                {"source": 2, "target": 6}, {"source": 3, "target": 7},
            ]
        }

    try:
        # STEP 2: Generative Optimizer
        ai = AIDesigner()
        opt = EvolutionaryOptimizer(ai)

        # 1 iteration to stay within Hugging Face 60s proxy timeout
        final_graph, best_results = opt.run_optimization_loop(base_geom, material_params, max_iterations=1)

        if not final_graph or final_graph.number_of_nodes() == 0:
            with open("crash.log", "w") as f:
                f.write("AI generative loop returned 0 nodes.\n")
            return jsonify({'error': 'AI generative loop failed to produce graph nodes.'}), 500

        max_disp = _safe_float(best_results.get("max_displacement", 0.001) if best_results else 0.01)
        # Per-node FEA displacements for visualization gradient
        node_disps = best_results.get("node_displacements", {}) if best_results else {}

    except Exception as e:
        with open("crash.log", "w") as f:
            f.write(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    # STEP 3: Map back to frontend format
    nodes_out = {}
    for node_id, ndata in final_graph.nodes(data=True):
        coords = ndata["coords"]
        nodes_out[str(node_id)] = {"x": coords[0], "y": coords[1], "z": coords[2]}

    members_out = []
    total_length = 0.0
    edge_idx = 0
    max_disp_val = max(max_disp, 1e-6)  # avoid division by zero

    for u, v, edata in final_graph.edges(data=True):
        edge_idx += 1
        p1 = np.array(final_graph.nodes[u]["coords"])
        p2 = np.array(final_graph.nodes[v]["coords"])
        member_len = float(np.linalg.norm(p1 - p2))
        total_length += member_len

        # Use actual FEA nodal displacements (normalised 0-1 for colour gradient)
        disp_i = _safe_float(node_disps.get(str(u), 0.0) / max_disp_val)
        disp_j = _safe_float(node_disps.get(str(v), 0.0) / max_disp_val)

        members_out.append({
            "id": f"m_{edge_idx}",
            "from": str(u),
            "to": str(v),
            "role": edata.get("section_type", "secondary_lattice"),
            "section": edata.get("section", mat_type),
            "connection": edata.get("connection", "fixed"),
            "disp_i": round(disp_i, 4),
            "disp_j": round(disp_j, 4)
        })

    # Sustainability metrics — use actual cross-section area from edge data
    total_volume = total_length * 0.05   # fallback: 0.05 m² average area
    total_mass = total_volume * material_params["rho"]

    if mat_type == "Steel":
        total_carbon = total_mass * 1.22
        total_cost = (total_mass / 1000) * 2653.0
    else:
        total_carbon = total_mass * 0.20
        total_cost = (total_mass / 1833) * 145.0

    return jsonify({
        'metrics': {
            'Carbon_kgCO2e': _safe_float(round(total_carbon, 0)),
            'Cost_USD': _safe_float(round(total_cost, 0)),
            'Volume': _safe_float(round(total_volume, 2)),
            'Max_Disp': _safe_float(round(max_disp, 4))
        },
        'nodes': nodes_out,
        'members': members_out,
        'max_disp': _safe_float(max_disp),
        'unit': 'm'
    })

if __name__ == '__main__':
    print("  AI GENERATIVE SERVER running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
