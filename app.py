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
                try:
                    # In a real scenario, we might upload the mesh here
                    # For now, we'll send the extracted geometry
                    payload = {
                        "material": mat_type,
                        "geometry": base_geom 
                    }
                    response = requests.post(f"{brain_url}/api/evaluate", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.variants = data["variants"]
                        # Default to balanced variant for initial display
                        balanced_variant = next((v for v in data["variants"] if v["name"] == "BALANCED"), data["variants"][0])
                        final_graph = None # Logic below will handle variants
                        best_results = balanced_variant["metrics"]
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
            status.update(label="AI Optimization Complete.", state="complete")
            
        if final_graph:
            # --- DASHBOARD METRICS ---
            st.header("Structural Performance Dashboard")
            
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
            if "variants" in st.session_state:
                v_names = [v["name"] for v in st.session_state.variants]
                selected_v_name = st.radio("Select Optimization Variant", v_names, horizontal=True)
                selected_variant = next(v for v in st.session_state.variants if v["name"] == selected_v_name)
                
                # Update metrics with selected variant data
                col1, col2, col3 = st.columns(3)
                col1.metric("Mass", f"{selected_variant['metrics']['Volume'] * rho:,.0f} kg")
                col2.metric("Carbon", f"{selected_variant['metrics']['Carbon_kgCO2e']:,.0f} kgCO2e")
                col3.metric("Cost", f"${selected_variant['metrics']['Cost_USD']:,.0f}")
                
                # Use variant nodes and members for plotting
                active_nodes = selected_variant["nodes"]
                active_members = selected_variant["members"]
            else:
                active_nodes = {str(n): {"x": coords[0], "y": coords[1], "z": coords[2]} for n, coords in final_graph.nodes(data="coords")}
                active_members = [{"from": str(u), "to": str(v), "disp_i": 0, "disp_j": 0} for u, v in final_graph.edges()]

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
                    # Use actual displacement if available
                    di = m.get("disp_i", 0)
                    dj = m.get("disp_j", 0)
                    colors.extend([di, dj, (di+dj)/2])
                else:
                    colors.extend([1, 1, 1])
                    
            if displacement_heatmap:
                line_marker = dict(color=colors, width=4, colorscale='Bluered', showscale=True, colorbar=dict(title="Displacement Heatmap"))
            else:
                line_marker = dict(color='black', width=3)
                
            fig = go.Figure(data=go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=line_marker,
                hoverinfo='none'
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Fragment Mono, monospace", color="rgba(255,255,255,0.85)"),
                scene=dict(
                    xaxis=dict(showbackground=False, visible=False),
                    yaxis=dict(showbackground=False, visible=False),
                    zaxis=dict(showbackground=False, visible=False),
                    camera=dict(projection=dict(type=projection))
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Auto-cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
