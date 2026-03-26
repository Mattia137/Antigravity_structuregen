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

# Section ladder: ordered lightest → heaviest by cross-sectional area
# Used to scale sections up/down for optimization variants
_STEEL_LADDER = [
    "W8x31", "IPE_300", "HEA_200", "HSS6x0.500",
    "W12x50", "W12x53", "Tubular_HSS_4x4x1/4",
    "HSS8x8x0.500", "HSS10x0.500",
    "W14x90", "W18x97", "W24x146",
    "W14x159", "HSS12x12x0.625", "HSS16x0.625", "W14x283"
]
_LADDER_INDEX = {s: i for i, s in enumerate(_STEEL_LADDER)}

def _scale_graph_sections(graph, steps: int):
    """
    Return a new graph with every steel section shifted by `steps` along the
    weight ladder. Positive = heavier (performance opt), negative = lighter
    (cost/carbon opt). Non-ladder sections are left unchanged.
    """
    import networkx as nx
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        sec = data.get("section", "")
        if sec in _LADDER_INDEX:
            new_idx = max(0, min(len(_STEEL_LADDER) - 1,
                                 _LADDER_INDEX[sec] + steps))
            G[u][v]["section"] = _STEEL_LADDER[new_idx]
    return G


def _run_fea(graph, material_params):
    """Build, load, and solve FEA on a graph. Returns results dict."""
    from src.fea_solver import FEASolver
    fea = FEASolver(graph, material_params)
    fea.build_model()
    fea.apply_loads()
    return fea.solve_and_evaluate()


def _graph_to_response(graph, fea_results, material_params, mat_type):
    """Convert a NetworkX graph + FEA results to the frontend member/node format."""
    max_disp = _safe_float(fea_results.get("max_displacement", 0.001) if fea_results else 0.01)
    node_disps = fea_results.get("node_displacements", {}) if fea_results else {}
    max_disp_val = max(max_disp, 1e-6)

    nodes_out = {}
    for node_id, ndata in graph.nodes(data=True):
        coords = ndata["coords"]
        nodes_out[str(node_id)] = {"x": coords[0], "y": coords[1], "z": coords[2]}

    members_out = []
    total_length = 0.0
    for edge_idx, (u, v, edata) in enumerate(graph.edges(data=True), start=1):
        p1 = np.array(graph.nodes[u]["coords"])
        p2 = np.array(graph.nodes[v]["coords"])
        member_len = float(np.linalg.norm(p1 - p2))
        total_length += member_len

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

    total_volume = total_length * 0.05  # 0.05 m² average cross-section area
    total_mass = total_volume * material_params["rho"]

    if mat_type == "Steel":
        total_carbon = total_mass * 1.22
        total_cost = (total_mass / 1000) * 2653.0
    else:
        total_carbon = total_mass * 0.20
        total_cost = (total_mass / 1833) * 145.0

    return {
        "nodes": nodes_out,
        "members": members_out,
        "max_disp": _safe_float(max_disp),
        "metrics": {
            "Carbon_kgCO2e": _safe_float(round(total_carbon, 0)),
            "Cost_USD": _safe_float(round(total_cost, 0)),
            "Volume": _safe_float(round(total_volume, 2)),
            "Max_Disp": _safe_float(round(max_disp, 4)),
            "Status": fea_results.get("status", "Unknown") if fea_results else "Unknown"
        }
    }


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
    Returns 3 optimization variants: cost/carbon, balanced, max performance.
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
    try:
        ge = GeometryEngine('mass-DEF.obj')
        all_verts = np.array(ge.extract_boundary_nodes())

        creases = ge.extract_primary_creases(angle_threshold_degrees=5.0)
        crease_set = set(tuple(sorted(e)) for e in creases["edges"])

        primary_nodes = [
            {"id": i, "x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
            for i, v in enumerate(all_verts)
        ]

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
        # STEP 2: AI Generative Design (1 iteration — stays within 60s proxy timeout)
        ai = AIDesigner()
        opt = EvolutionaryOptimizer(ai)
        base_graph, base_results = opt.run_optimization_loop(base_geom, material_params, max_iterations=1)

        if not base_graph or base_graph.number_of_nodes() == 0:
            with open("crash.log", "w") as f:
                f.write("AI generative loop returned 0 nodes.\n")
            return jsonify({'error': 'AI generative loop failed to produce graph nodes.'}), 500

        # STEP 3: Generate 3 variants programmatically
        # Variant A — Minimum Cost & Carbon: downgrade sections 2 steps
        graph_cost = _scale_graph_sections(base_graph, steps=-2)
        results_cost = _run_fea(graph_cost, material_params)

        # Variant B — Balanced (base AI design, no section change)
        results_balanced = base_results

        # Variant C — Minimum Displacement: upgrade sections 2 steps
        graph_perf = _scale_graph_sections(base_graph, steps=+2)
        results_perf = _run_fea(graph_perf, material_params)

    except Exception as e:
        with open("crash.log", "w") as f:
            f.write(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    # STEP 4: Build response for all 3 variants
    variant_cost    = _graph_to_response(graph_cost,    results_cost,     material_params, mat_type)
    variant_balanced = _graph_to_response(base_graph,   results_balanced, material_params, mat_type)
    variant_perf    = _graph_to_response(graph_perf,    results_perf,     material_params, mat_type)

    variant_cost["name"]     = "MIN_COST"
    variant_balanced["name"] = "BALANCED"
    variant_perf["name"]     = "MIN_DISP"

    return jsonify({
        'variants': [variant_cost, variant_balanced, variant_perf],
        # Legacy flat fields — mirrors balanced variant for backwards compat
        'nodes':   variant_balanced['nodes'],
        'members': variant_balanced['members'],
        'metrics': variant_balanced['metrics'],
        'max_disp': variant_balanced['max_disp'],
        'unit': 'm'
    })

if __name__ == '__main__':
    print("  AI GENERATIVE SERVER running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
