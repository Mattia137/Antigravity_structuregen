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
from src.structural_rules_bridge import (
    compute_mesh_descriptors,
    run_code_checks,
    rules_digest_for_prompt,
)

app = Flask(__name__, static_folder='static')
CORS(app)


# ── Response builder ──────────────────────────────────────────────────────────

def _graph_to_response(graph, fea_results, material_params, mat_type, mesh_desc=None):
    """Convert a NetworkX graph + FEA results to the frontend member/node format."""
    max_disp = _safe_float(fea_results.get("max_displacement", 0.001) if fea_results else 0.01)
    node_disps = fea_results.get("node_displacements", {}) if fea_results else {}
    max_disp_val = max(max_disp, 1e-6)

    nodes_out = {}
    for node_id, ndata in graph.nodes(data=True):
        coords = ndata["coords"]
        nodes_out[str(node_id)] = {
            "x": coords[0], "y": coords[1], "z": coords[2],
            "connection_type": ndata.get("connection_type", "welded")
        }

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
            "from": str(u), "to": str(v),
            "role": edata.get("section_type", "secondary_lattice"),
            "section": edata.get("section", mat_type),
            "connection": edata.get("connection", "fixed"),
            "typology": edata.get("typology", "welded"),
            "disp_i": round(disp_i, 4),
            "disp_j": round(disp_j, 4),
        })

    # Sustainability metrics
    total_volume = total_length * 0.05   # 0.05 m² average cross-section area
    total_mass   = total_volume * material_params["rho"]

    if mat_type == "Steel":
        total_carbon = total_mass * 1.22
        total_cost   = (total_mass / 1000) * 2653.0
    else:
        total_carbon = total_mass * 0.20
        total_cost   = (total_mass / 1833) * 145.0

    # Code check summary for this variant
    code_report = {}
    if mesh_desc:
        code_report = run_code_checks(fea_results or {}, mesh_desc, graph)

    return {
        "nodes":   nodes_out,
        "members": members_out,
        "max_disp": _safe_float(max_disp),
        "metrics": {
            "Carbon_kgCO2e": _safe_float(round(total_carbon, 0)),
            "Cost_USD":      _safe_float(round(total_cost, 0)),
            "Volume":        _safe_float(round(total_volume, 2)),
            "Max_Disp":      _safe_float(round(max_disp, 4)),
            "Status":        fea_results.get("status", "Unknown") if fea_results else "Unknown",
            "Code_Overall":  code_report.get("overall", "unknown"),
            "Drift_DCR":     _safe_float(code_report.get("drift_DCR", 0.0)),
        },
        "code_checks": code_report.get("checks", []),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

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
            'Steel':    {'E': 29000, 'nu': 0.3, 'rho': 0.283, 'Strength': 50},
            'Concrete': {'E': 4000,  'nu': 0.2, 'rho': 0.086, 'Strength': 5}
        }
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """
    Full generative + FEA + code-check + 3-variant pipeline.

    Steps:
      1. Extract mesh geometry (vertices + classified edges)
      2. Compute mesh descriptors + select structural system (rules bridge)
      3. One Gemini call → base design
      4. PyNite FEA + code checks (ASCE 7-22 / IBC 2024 / AISC 360-22)
      5. Revision loop if code checks fail (up to max_iterations)
      6. Section-scale to MIN_COST (-2) and MIN_DISP (+2); FEA on each
      7. Return all 3 variants + code check results
    """
    try:
        data = request.json or {}
        mat_type = data.get('material', 'Steel')

        material_params = {
            "type": mat_type,
            "E":   200e9 if mat_type == 'Steel' else 30e9,
            "nu":  0.3   if mat_type == 'Steel' else 0.2,
            "rho": 7850.0 if mat_type == 'Steel' else 2400.0,
            "G":   77e9  if mat_type == 'Steel' else 12e9,
            "Fy":  350e6 if mat_type == 'Steel' else 40e6,
        }

        # ── STEP 1: Geometry Extraction ───────────────────────────────────────
        mesh_path = 'mass-DEF.obj' # fallback
        mesh_b64 = data.get('mesh_b64')
        if mesh_b64:
            import base64
            mesh_path = 'temp_input.obj'
            with open(mesh_path, 'wb') as f:
                f.write(base64.b64decode(mesh_b64))
        
        ge = GeometryEngine(mesh_path)
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

        peak_points = ge.get_max_height_points()

        print(f"Geometry ({mesh_path}): {len(primary_nodes)} nodes, {len(primary_edges)} edges")

        # ── STEP 2: Mesh Descriptors + System Selection ───────────────────────
        mesh_desc = compute_mesh_descriptors(ge)
        rules_digest = rules_digest_for_prompt(mesh_desc)

        if mesh_desc:
            print(f"System: {mesh_desc.get('lateral_system')} | "
                  f"H/W={mesh_desc.get('H_W_ratio',0):.2f} | "
                  f"{mesh_desc.get('aspect_category')}")
        else:
            print("Mesh descriptors unavailable (rules module import failed).")

        base_geom = {
            "primary_nodes":  primary_nodes,
            "primary_edges":  primary_edges,
            "peak_points":    peak_points,
            "sqft_data":      ge.slice_mesh_horizontally(),
            "rules_digest":   rules_digest,
            "mesh_desc":      mesh_desc,
            "bounds": {
                "x_min": float(all_verts[:, 0].min()), "x_max": float(all_verts[:, 0].max()),
                "y_min": float(all_verts[:, 1].min()), "y_max": float(all_verts[:, 1].max()),
                "z_min": float(all_verts[:, 2].min()), "z_max": float(all_verts[:, 2].max()),
            },
        }

        # ── STEP 3 + 4 + 5: AI Design → FEA → Code Checks → Revision ────────
        ai  = AIDesigner()
        opt = EvolutionaryOptimizer(ai)

        # max_iterations=1 to stay within 60s HF proxy timeout.
        # Increase locally if needed.
        base_graph, base_results = opt.run_optimization_loop(
            base_geom, material_params, max_iterations=1
        )

        if not base_graph or base_graph.number_of_nodes() == 0:
            return jsonify({'error': 'AI pipeline produced no structural nodes.'}), 500

        # ── STEP 6: Build 3 variants via section scaling + FEA ────────────────
        variants_raw = opt.build_three_variants(base_graph, base_results, material_params)

        # ── STEP 7: Format for frontend ───────────────────────────────────────
        output_variants = []
        for v in variants_raw:
            resp = _graph_to_response(v["graph"], v["fea"], material_params, mat_type, mesh_desc)
            resp["name"] = v["goal"]
            output_variants.append(resp)

        # Default display: BALANCED (index 1)
        primary = output_variants[1]

        return jsonify({
            'variants': output_variants,
            # Legacy flat fields for backwards compatibility
            'nodes':    primary['nodes'],
            'members':  primary['members'],
            'metrics':  primary['metrics'],
            'max_disp': primary['max_disp'],
            'unit':     'm',
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        with open("crash.log", "w") as f:
            f.write(error_msg)
        print(f"evaluate() error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("  AI GENERATIVE SERVER running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
