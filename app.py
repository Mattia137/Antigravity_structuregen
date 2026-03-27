import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time
import os

from src.geometry_engine import GeometryEngine
from src.ai_designer import AIDesigner
from src.optimizer import EvolutionaryOptimizer
from src.fea_solver import FEASolver

st.set_page_config(page_title="Generative Structural Exoskeleton", layout="wide")

try:
    with open("static/style.css", "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# --- UI PARAMETER BLOCK (MANDATORY) ---
with st.sidebar:
    st.header("Parameter Block")
    st.subheader("Material Properties")
    mat_type = st.selectbox("Material Type", ["Steel", "Concrete"])
    
    if mat_type == "Steel":
        E_mod = st.number_input("Young's Modulus (E) [Pa]", value=200e9, format="%e")
        Fy = st.number_input("Yield Strength (Fy) [Pa]", value=350e6, format="%e")
        rho = st.number_input("Density (rho) [kg/m^3]", value=7850.0)
        alpha = st.number_input("Thermal Expansion (\u03B1)", value=1.2e-5, format="%e")
        nu = st.number_input("Poisson's Ratio (\u03BD)", value=0.3)
    else:
        E_mod = st.number_input("Young's Modulus (E) [Pa]", value=30e9, format="%e")
        Fy = st.number_input("Compressive Strength (f'c) [Pa]", value=40e6, format="%e")
        rho = st.number_input("Density (rho) [kg/m^3]", value=2400.0)
        alpha = st.number_input("Thermal Expansion (\u03B1)", value=1.0e-5, format="%e")
        nu = st.number_input("Poisson's Ratio (\u03BD)", value=0.2)
        
    gravity_toggle = st.radio("Gravity", [r"Earth (9.81 m/s^2)", r"Mars (3.71 m/s^2)"])
    gravity_val = 9.81 if "Earth" in gravity_toggle else 3.71
    
    st.subheader("Camera & View Parameters")
    projection = st.selectbox("Projection", ["perspective", "orthographic"])
    render_mode = st.radio("Render Mode", ["Wireframe", "Solid"])
    displacement_heatmap = st.checkbox("Toggle Displacement Heatmap", value=True)

# Build parameter dictionary
material_params = {
    "type": mat_type,
    "E": E_mod,
    "nu": nu,
    "rho": rho,
    "alpha": alpha,
    "Fy": Fy,
    "G": E_mod / (2 * (1 + nu))
}

with st.sidebar:
    st.header("Pipeline Configuration")
    use_remote = st.toggle("Use Remote Brain (Vercel/HuggingFace Proxy)", value=False)
    brain_url = st.text_input("Brain API URL", value="http://localhost:5000")

st.title("Generative Engineering Agent: Structural Evolution")
st.markdown("Ingests massing meshes, applies generative logic for load-paths, and optimizes via PyNite & Gemini.")

uploaded_mesh = st.file_uploader("Upload Massing Mesh (.obj / .stl)", type=["obj", "stl"])

if uploaded_mesh is not None:
    temp_path = f"temp_mesh_{uploaded_mesh.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_mesh.getbuffer())
        
    st.success("Mesh Uploaded successfully. 1:1 Scale Processing.")
    
    if st.button("Initialize Generative Design Pipeline"):
        with st.status("Executing Phase 2: Geometry Extraction...", expanded=True) as status:
            ge = GeometryEngine(temp_path)
            nodes = ge.extract_boundary_nodes()
            creases = ge.extract_primary_creases()
            sqft_data = ge.slice_mesh_horizontally()
            base_geom = {
                "vertices": nodes,
                "creases": creases,
                "sqft_data": sqft_data
            }
            status.update(label="Geometry Extraction Complete.", state="complete")
            
        with st.status("Executing AI Evolutionary Optimization...", expanded=True) as status:
            if use_remote:
                import requests
                import base64
                try:
                    # Encode the specific uploaded mesh as base64 to send to the remote brain
                    with open(temp_path, "rb") as f:
                        mesh_b64 = base64.b64encode(f.read()).decode('utf-8')

                    payload = {
                        "material": mat_type,
                        "geometry": base_geom,
                        "mesh_b64": mesh_b64
                    }
                    response = requests.post(f"{brain_url}/api/evaluate", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.variants = data["variants"]
                        # Default to DISPLACEMENT variant for initial display
                        default_variant = next((v for v in data["variants"] if v["name"] == "DISPLACEMENT"), data["variants"][0])
                        final_graph = None # Logic below will handle variants
                        best_results = default_variant["metrics"]
                        st.success("Remote AI generation successful.")
                    else:
                        st.error(f"Remote Brain Error: {response.text}")
                        final_graph = None
                        best_results = None
                except Exception as e:
                    st.error(f"Failed to connect to Remote Brain: {e}")
                    final_graph = None
                    best_results = None
            else:
                ai = AIDesigner()
                optimizer = EvolutionaryOptimizer(ai)
                final_graph, best_results = optimizer.run_optimization_loop(base_geom, material_params, max_iterations=3)
                # Store local optimization variants in session state if available
                if final_graph and hasattr(final_graph, "graph") and "variants" in final_graph.graph:
                    st.session_state.variants = final_graph.graph["variants"]

            status.update(label="AI Optimization Complete.", state="complete")
            
        if final_graph or "variants" in st.session_state:
            # --- DASHBOARD METRICS ---
            st.header("Structural Performance Dashboard")
            col1, col2, col3 = st.columns(3)
            
            if final_graph and not "variants" in st.session_state:
                # Simple volume/mass heuristic for the prototype
                # Primary structural elements
                total_length = 0.0
                for u, v, data in final_graph.edges(data=True):
                    coord_u = np.array(final_graph.nodes[u]["coords"])
                    coord_v = np.array(final_graph.nodes[v]["coords"])
                    total_length += np.linalg.norm(coord_u - coord_v)

                total_volume = total_length * 0.05 # average section area approx 0.05 m^2
                total_mass = total_volume * rho # kg
                
                # Knowledge Base Mapping
                if mat_type == "Steel":
                    total_carbon = total_mass * 1.22
                    total_cost = (total_mass / 1000) * 2653.0
                else:
                    total_carbon = total_mass * 0.20
                    total_cost = (total_mass / 1833) * 145.0

                col1.metric("Total Mass", f"{total_mass:,.0f} kg")
                col2.metric("Embodied Carbon (A1-A3)", f"{total_carbon:,.0f} kgCO2e")
                col3.metric("Material Cost", f"${total_cost:,.2f}")
                
            # --- PLOTLY VISUALIZATION ---
            st.header("3D Rendering")
            
            # Variant Selection (if available)
            selected_v_name = "DISPLACEMENT" # default fallback
            if "variants" in st.session_state and st.session_state.variants:
                v_names = [v["name"] for v in st.session_state.variants]
                selected_v_name = st.radio("Select Optimization Variant", v_names, horizontal=True)
                selected_variant = next(v for v in st.session_state.variants if v["name"] == selected_v_name)
                
                # Update metrics with selected variant data
                col1.metric("Mass", f"{selected_variant['metrics']['Volume'] * rho:,.0f} kg")
                col2.metric("Carbon", f"{selected_variant['metrics']['Carbon_kgCO2e']:,.0f} kgCO2e")
                col3.metric("Cost", f"${selected_variant['metrics']['Cost_USD']:,.0f}")
                
                # Use variant nodes and members for plotting
                active_nodes = selected_variant["nodes"]
                active_members = selected_variant["members"]
                if not final_graph and "graph" in selected_variant:
                    final_graph = selected_variant["graph"]
            else:
                active_nodes = {}
                for n, data in final_graph.nodes(data=True):
                    coords = data["coords"]
                    active_nodes[str(n)] = {
                        "x": coords[0],
                        "y": coords[1],
                        "z": coords[2],
                        "connection_type": data.get("connection_type", "welded")
                    }
                active_members = []
                for u, v, m_data in final_graph.edges(data=True):
                    active_members.append({
                        "from": str(u),
                        "to": str(v),
                        "disp_i": 0,
                        "disp_j": 0,
                        "section": m_data.get("section", "unknown"),
                        "typology": m_data.get("typology", "unknown")
                    })

            # --- RESULTS TABLE & DIAGRAM ---
            if "variants" in st.session_state and st.session_state.variants:
                st.header("Optimization Comparison")
                colA, colB = st.columns([1, 1.5])
                
                with colA:
                    st.subheader("Performance Metrics Table")
                    import pandas as pd
                    table_data = []
                    for v in st.session_state.variants:
                        m = v["metrics"]
                        table_data.append({
                            "Variant": v["name"],
                            "Cost ($)": f"${m['Cost_USD']:,.0f}",
                            "Carbon (kgCO2e)": f"{m['Carbon_kgCO2e']:,.0f}",
                            "Max Disp (m)": f"{m['Max_Disp']:.4f}"
                        })
                    st.table(pd.DataFrame(table_data))
                
                with colB:
                    st.subheader("Comparative Pareto Analysis")
                    # Radar or Scatter comparison
                    v_names = [v["name"] for v in st.session_state.variants]
                    v_costs = [v["metrics"]["Cost_USD"] for v in st.session_state.variants]
                    v_carbon = [v["metrics"]["Carbon_kgCO2e"] for v in st.session_state.variants]
                    v_disp = [v["metrics"]["Max_Disp"] for v in st.session_state.variants]
                    
                    # Normalized data for radar
                    n_costs = [c / max(v_costs) for c in v_costs]
                    n_carbon = [ca / max(v_carbon) for ca in v_carbon]
                    n_disp = [d / max(v_disp) for d in v_disp]
                    
                    fig_diag = go.Figure()
                    for idx, name in enumerate(v_names):
                        fig_diag.add_trace(go.Scatterpolar(
                            r=[n_costs[idx], n_carbon[idx], n_disp[idx]],
                            theta=['Cost', 'Carbon', 'Displacement'],
                            fill='toself',
                            name=name
                        ))
                    fig_diag.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
                    st.plotly_chart(fig_diag, use_container_width=True)

                st.subheader("Variant Typology Distribution")
                colC, colD = st.columns(2)

                with colC:
                    node_conns = [n.get("connection_type", "unknown") for n in active_nodes.values()]
                    conn_counts = {c: node_conns.count(c) for c in set(node_conns)}
                    fig_conn = go.Figure(data=[go.Pie(labels=list(conn_counts.keys()), values=list(conn_counts.values()), hole=.3)])
                    fig_conn.update_layout(title="Node Connections", margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_conn, use_container_width=True)

                with colD:
                    member_secs = [m.get("section", "unknown") for m in active_members]
                    sec_counts = {s: member_secs.count(s) for s in set(member_secs)}
                    fig_sec = go.Figure(data=[go.Pie(labels=list(sec_counts.keys()), values=list(sec_counts.values()), hole=.3)])
                    fig_sec.update_layout(title="Member Cross-Sections", margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_sec, use_container_width=True)

            # Build node displacements mapping if heatmap is active
            node_displacements = None
            if displacement_heatmap:
                node_displacements = {}
                for m in active_members:
                    node_displacements[m["from"]] = m.get("disp_i", 0.0)
                    node_displacements[m["to"]] = m.get("disp_j", 0.0)

            # Re-generate solid structure for the selected variant
            # Since the graph object might not be cleanly saved in session state for Streamlit,
            # we check if we have the active graph.
            # In our local run_optimization_loop, the variants are saved to final_graph.graph["variants"]
            active_graph = None
            if hasattr(final_graph, "graph") and "variants" in final_graph.graph:
                v_selected = next(v for v in final_graph.graph["variants"] if v["name"] == (selected_v_name if "variants" in st.session_state else "DISPLACEMENT"))
                active_graph = v_selected.get("graph")

            if active_graph is None:
                active_graph = final_graph

            from src.geometry_engine import GeometryEngine
            import trimesh
            import plotly.colors as pc
            
            solid_mesh = None
            # Extract displacements locally if we have active_graph
            node_disp_for_mesh = node_displacements

            solid_mesh = GeometryEngine.generate_solid_structure(active_graph, node_displacements=node_disp_for_mesh)

            fig = go.Figure()

            if solid_mesh and isinstance(solid_mesh, trimesh.Trimesh):
                vertices = solid_mesh.vertices
                faces = solid_mesh.faces
                x = vertices[:, 0].tolist()
                y = vertices[:, 1].tolist()
                z = vertices[:, 2].tolist()
                i = faces[:, 0].tolist()
                j = faces[:, 1].tolist()
                k = faces[:, 2].tolist()
                
                # If displacement_heatmap, solid_mesh generated colors. Extract them
                if displacement_heatmap and hasattr(solid_mesh.visual, 'vertex_colors') and len(solid_mesh.visual.vertex_colors) > 0:
                    v_colors = solid_mesh.visual.vertex_colors
                    # convert rgba array to hex strings
                    color_vals = [f"rgb({c[0]}, {c[1]}, {c[2]})" for c in v_colors]
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        vertexcolor=color_vals,
                        flatshading=False,
                        hoverinfo='none'
                    ))
                else:
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        color='black' if not render_mode == "Solid" else 'lightgrey',
                        flatshading=True,
                        hoverinfo='none'
                    ))
            else:
                # Fallback to lines if solid generation fails
                edge_x = []
                edge_y = []
                edge_z = []
                colors = []
                
                for m in active_members:
                    u, v = m["from"], m["to"]
                    cu = [active_nodes[u]["x"], active_nodes[u]["y"], active_nodes[u]["z"]]
                    cv = [active_nodes[v]["x"], active_nodes[v]["y"], active_nodes[v]["z"]]
                    edge_x.extend([cu[0], cv[0], None])
                    edge_y.extend([cu[1], cv[1], None])
                    edge_z.extend([cu[2], cv[2], None])

                    if displacement_heatmap:
                        di = m.get("disp_i", 0)
                        dj = m.get("disp_j", 0)
                        colors.extend([di, dj, (di+dj)/2])

                if displacement_heatmap:
                    line_marker = dict(color=colors, width=4, colorscale='Bluered', showscale=True)
                else:
                    line_marker = dict(color='black', width=3)

                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=line_marker,
                    hoverinfo='none'
                ))
            
            # Calculate Dynamic Camera Center
            if active_nodes:
                xs = [n["x"] for n in active_nodes.values()]
                ys = [n["y"] for n in active_nodes.values()]
                zs = [n["z"] for n in active_nodes.values()]
                center_x = (max(xs) + min(xs)) / 2
                center_y = (max(ys) + min(ys)) / 2
                center_z = (max(zs) + min(zs)) / 2
                max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
            else:
                center_x, center_y, center_z, max_dim = 0, 0, 0, 10

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Fragment Mono, monospace", color="rgba(255,255,255,0.85)"),
                scene=dict(
                    xaxis=dict(showbackground=False, visible=False),
                    yaxis=dict(showbackground=False, visible=False),
                    zaxis=dict(showbackground=False, visible=False),
                    camera=dict(
                        projection=dict(type=projection),
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1) # Z-up
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=800
            )
            
            # Adjust camera center and eye based on mesh bounds
            fig.update_scenes(camera_center=dict(x=0, y=0, z=0)) 
            # Note: since we normalized the mesh in GeometryEngine to be centered at 0,0 
            # and min_z=0, the camera center (0,0,0) is near the base.
            # We shift it up to the middle of the building.
            fig.update_layout(scene_camera_center=dict(x=0, y=0, z=center_z/(max_dim if max_dim > 0 else 1)))

            st.plotly_chart(fig, use_container_width=True)
            
            # Auto-cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
