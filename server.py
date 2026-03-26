from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import json, os, sys

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

import config
from struct_generator import generate_structure
from optimization import evaluate_model

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Serve the main page ──
@app.route('/')
def index():
    return send_file('static/index.html')

# ── Serve the OBJ file directly ──
@app.route('/mesh/<path:filename>')
def serve_mesh(filename):
    return send_from_directory('.', filename)

# ── API: Get available materials & sections ──
@app.route('/api/config')
def get_config():
    return jsonify({
        'materials': config.MATERIALS,
        'sections': config.SECTIONS,
        'defaults': config.DEFAULTS,
        'seismic': config.SEISMIC
    })

# ── API: Generate structure + run FEA ──
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    mat = data.get('material', 'Steel')
    num_floors = int(data.get('num_floors', 6))
    floor_height_input = float(data.get('floor_height', 3.66))
    unit = data.get('unit', 'm')  # 'm' or 'ft'
    custom_props = data.get('custom_props', None)
    member_overrides = data.get('member_overrides', None)

    # Conversion factors: user unit → inches (PyNite internal)
    if unit == 'm':
        to_inches = 39.3701
    else:  # ft
        to_inches = 12.0

    # Convert to inches for PyNite
    floor_height_in = floor_height_input * to_inches
    # Bay size: derive from a reasonable fraction of floor height
    bay_size_in = floor_height_input * to_inches * 1.5  # ~1.5x floor height

    # Handle file upload or default mesh
    # If the user uploaded a file, it would be passed differently, but for now we expect the client
    # to either send the file via multipart form-data, or we just rely on the fallback mass-DEF.obj
    # Let's load the default for now (or a temp file if uploaded)
    import trimesh
    from mesh_utils import load_building_mesh
    mesh, vertices_list, lines_list = load_building_mesh(filepath='mass-DEF.obj') # Fallback or actual
    # Scale mesh vertices to inches for PyNite context
    mesh.apply_scale(to_inches)
    
    # Also scale the raw vertices list
    scaled_vertices = [(x*to_inches, y*to_inches, z*to_inches) for x, y, z in vertices_list]

    model, members, nodes_dict = generate_structure(
        scaled_vertices, lines_list,
        num_floors, floor_height_in,
        bay_size=bay_size_in,
        selected_material=mat,
        custom_props=custom_props,
        member_overrides=member_overrides
    )

    metrics, node_disps = evaluate_model(model)
    if metrics is None:
        return jsonify({'error': 'Analysis failed'}), 500

    # Convert node positions from inches back to display unit
    from_inches = 1.0 / to_inches
    nodes_out = {}
    for name, node in model.nodes.items():
        nodes_out[name] = {
            'x': node.X * from_inches,
            'y': node.Y * from_inches,
            'z': node.Z * from_inches
        }

    # Convert volume and displacement to display units
    vol_display = metrics['Volume_in3'] * (from_inches ** 3)
    disp_display = metrics['Max_Disp_in'] * from_inches

    members_out = []
    max_d = max(metrics['Max_Disp_in'], 0.001)
    for m in members:
        d_i = node_disps.get(m['From'], 0) / max_d if node_disps else 0
        d_j = node_disps.get(m['To'], 0) / max_d if node_disps else 0
        import math
        if math.isnan(d_i) or math.isinf(d_i): d_i = 0
        if math.isnan(d_j) or math.isinf(d_j): d_j = 0
        members_out.append({
            'id': m['MemberID'],
            'from': m['From'],
            'to': m['To'],
            'role': m['StructuralRole'],
            'section': m['SectionType'],
            'disp_i': d_i,
            'disp_j': d_j
        })

    return jsonify({
        'metrics': {
            'Carbon_kgCO2e': metrics['Carbon_kgCO2e'],
            'Cost_USD': metrics['Cost_USD'],
            'Volume': round(vol_display, 2),
            'Max_Disp': round(disp_display, 4)
        },
        'nodes': nodes_out,
        'members': members_out,
        'max_disp': disp_display,
        'unit': unit
    })

if __name__ == '__main__':
    print("  ANTIGRAVITY_ANALYSIS server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
