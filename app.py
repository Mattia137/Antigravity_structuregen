import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import config
from mesh_utils import load_building_mesh, slice_mesh_and_get_floorplates, extract_plotly_mesh
from struct_generator import generate_structure
from optimization import evaluate_model

# ───────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & FUI AESTHETIC (Matching HYPEROBJECT_environment)
# ───────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="ANTIGRAVITY_ANALYSIS", page_icon="⬡")

FUI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fragment+Mono:ital@0;1&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Fragment Mono', monospace !important;
    color: rgba(255,255,255,0.85) !important;
}

/* ── App Background ── */
.stApp {
    background: #000000 !important;
}
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0;
    box-shadow: inset 0 0 120px 60px rgba(30,90,255,0.07);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.8) !important;
}

/* ── Headers ── */
h1, h2, h3, h4 {
    font-family: 'Fragment Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.25em !important;
    font-weight: 400 !important;
    color: rgba(255,255,255,0.9) !important;
}
h1 { font-size: 14px !important; }
h2, h3 { font-size: 11px !important; }
h4 { font-size: 9px !important; }

/* ── Body Text ── */
p, span, label, .stMarkdown, div {
    font-size: 9px !important;
    letter-spacing: 0.12em !important;
}

/* ── Inputs ── */
input, select, textarea, .stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #fff !important;
    font-family: 'Fragment Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.1em !important;
}

/* ── Ghost Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Fragment Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    padding: 12px 30px !important;
    border-radius: 0px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,1) !important;
    color: #000 !important;
}
/* Override the red primary button */
.stButton > button[kind="primary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    color: rgba(255,255,255,0.9) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #fff !important;
    color: #000 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 10px 15px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 6px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    opacity: 0.5 !important;
}
[data-testid="stMetricValue"] {
    font-size: 16px !important;
    letter-spacing: 0.15em !important;
}

/* ── Dataframe / Data Editor (DARK) ── */
.stDataFrame, [data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stDataFrame"] iframe {
    filter: invert(1) hue-rotate(180deg) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.02) !important;
    border-radius: 0 !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 8px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 0 !important;
    font-family: 'Fragment Mono', monospace !important;
    font-size: 8px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
[data-testid="stFileUploader"] small {
    color: rgba(255,255,255,0.3) !important;
}

/* ── Horizontal Rules ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

/* ── Info/Success/Alert Boxes ── */
.stAlert {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.7) !important;
    font-size: 8px !important;
    letter-spacing: 0.15em !important;
    border-radius: 0 !important;
}

/* ── Slider ── */
.stSlider > div > div {
    color: rgba(255,255,255,0.7) !important;
}

/* ── Tabs (if used) ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Fragment Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    font-size: 8px !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
"""
st.markdown(FUI_CSS, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────
# 2. TITLE BLOCK
# ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 15px 0;">
    <div style="font-size:6px; letter-spacing:0.5em; opacity:0.3; text-transform:uppercase; margin-bottom:8px;">SCI-ARC / DS 2GBX</div>
    <div style="font-size:16px; letter-spacing:0.5em; text-transform:uppercase; color:rgba(255,255,255,0.9);">ANTIGRAVITY _ ANALYSIS</div>
    <div style="font-size:7px; letter-spacing:0.3em; opacity:0.35; margin-top:6px; text-transform:uppercase;">LABC / ASCE 7-22 / LEED Sustainability</div>
</div>
""", unsafe_allow_html=True)

if 'iterations' not in st.session_state:
    st.session_state.iterations = []
if 'member_overrides' not in st.session_state:
    st.session_state.member_overrides = {}

# ───────────────────────────────────────────────────────────────────────
# 3. SIDEBAR
# ───────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-size:7px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.4; margin-bottom:15px;">GLOBAL PARAMETER BLOCK</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size:7px; letter-spacing:0.3em; text-transform:uppercase; opacity:0.5; margin-bottom:5px;">MESH IMPORT</div>', unsafe_allow_html=True)
uploaded_mesh = st.sidebar.file_uploader("UPLOAD .OBJ / .STL", type=["obj", "stl"], label_visibility="collapsed")

st.sidebar.markdown('<div style="font-size:7px; letter-spacing:0.3em; text-transform:uppercase; opacity:0.5; margin-bottom:5px;">GEOMETRY</div>', unsafe_allow_html=True)
floor_height = st.sidebar.number_input("FLOOR_HEIGHT (IN)", value=144, step=12)
num_floors = st.sidebar.slider("NUM_FLOORS", min_value=1, max_value=25, value=6)

st.sidebar.markdown('<div style="font-size:7px; letter-spacing:0.3em; text-transform:uppercase; opacity:0.5; margin-top:15px; margin-bottom:5px;">MATERIAL</div>', unsafe_allow_html=True)
mat_choice = st.sidebar.selectbox("PRESET", ["Steel", "Concrete", "Wood"])
defaults = config.MATERIALS[mat_choice]
E = st.sidebar.number_input("E_MODULUS (KSI)", value=float(defaults['E']))
nu = st.sidebar.number_input("POISSON", value=float(defaults['nu']))
rho = st.sidebar.number_input("DENSITY (LB/IN³)", value=float(defaults['rho']))
strength = st.sidebar.number_input("STRENGTH (KSI)", value=float(defaults['Strength']))
custom_props = {'E': E, 'nu': nu, 'rho': rho, 'Strength': strength}

st.sidebar.markdown('<div style="font-size:7px; letter-spacing:0.3em; text-transform:uppercase; opacity:0.5; margin-top:15px; margin-bottom:5px;">SEISMIC</div>', unsafe_allow_html=True)
s_ds = st.sidebar.number_input("S_DS (G)", value=float(config.SEISMIC['S_DS']))
config.SEISMIC['S_DS'] = s_ds

# ───────────────────────────────────────────────────────────────────────
# 4. STRUCTURAL DATAFRAME
# ───────────────────────────────────────────────────────────────────────
base_model, raw_members, nodes_dict = generate_structure(
    num_floors, floor_height,
    selected_material=mat_choice,
    custom_props=custom_props,
    member_overrides=st.session_state.member_overrides
)

col_left, col_right = st.columns([1.3, 0.7])

with col_left:
    st.markdown('<div style="font-size:7px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.4; margin-bottom:8px;">MEMBER SCHEDULE</div>', unsafe_allow_html=True)
    df_mem = pd.DataFrame(raw_members)
    available_sections = list(config.SECTIONS[mat_choice].keys())
    edited_df = st.data_editor(
        df_mem,
        column_config={
            "SectionType": st.column_config.SelectboxColumn("SECTION", options=available_sections, required=True),
            "MemberID": st.column_config.TextColumn("ID", disabled=True),
            "From": st.column_config.TextColumn("N_I", disabled=True),
            "To": st.column_config.TextColumn("N_J", disabled=True),
            "StructuralRole": st.column_config.TextColumn("ROLE", disabled=True)
        },
        use_container_width=True, hide_index=True, height=280
    )
    for idx, row in edited_df.iterrows():
        mid = row['MemberID']
        if row['SectionType'] != df_mem.iloc[idx]['SectionType']:
            st.session_state.member_overrides[mid] = row['SectionType']

with col_right:
    st.markdown('<div style="font-size:7px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.4; margin-bottom:8px;">ACTIONS</div>', unsafe_allow_html=True)
    if st.button("EVALUATE _ DESIGN", type="primary"):
        with st.spinner("ANALYZING..."):
            metrics, node_disps = evaluate_model(base_model)
            if metrics:
                metrics['Iteration'] = f"OPT_{len(st.session_state.iterations)+1}"
                st.session_state.iterations.append({
                    'Metrics': metrics, 'Disps': node_disps,
                    'Model': base_model, 'Members': raw_members
                })
    if st.button("CLEAR _ MEMORY"):
        st.session_state.iterations = []
        st.session_state.member_overrides = {}
        st.rerun()

# ───────────────────────────────────────────────────────────────────────
# 5. PLOTLY SCENE TEMPLATE (Dark FUI)
# ───────────────────────────────────────────────────────────────────────
SCENE_LAYOUT = dict(
    bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showticklabels=False, title=''),
    yaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showticklabels=False, title=''),
    zaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showticklabels=False, title='')
)
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, b=0, t=0), height=500,
    scene=SCENE_LAYOUT, scene_camera=config.CAMERA_CONFIG,
    font=dict(family='Fragment Mono, monospace', size=8, color='rgba(255,255,255,0.5)')
)

# ───────────────────────────────────────────────────────────────────────
# 6. VISUALIZER DASHBOARDS
# ───────────────────────────────────────────────────────────────────────
st.markdown('<hr style="border-color:rgba(255,255,255,0.04); margin: 25px 0;">', unsafe_allow_html=True)

if len(st.session_state.iterations) > 0:
    latest = st.session_state.iterations[-1]
    last_model = latest['Model']
    last_members = latest['Members']
    node_disps = latest['Disps']

    # Metrics row
    m = latest['Metrics']
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("CARBON (KGCO2E)", f"{m['Carbon_kgCO2e']:,.0f}")
    mcol2.metric("COST (USD)", f"${m['Cost_USD']:,.0f}")
    mcol3.metric("VOLUME (IN³)", f"{m['Volume_in3']:,.0f}")
    mcol4.metric("MAX DRIFT (IN)", f"{m['Max_Disp_in']:.3f}")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.04); margin: 15px 0;">', unsafe_allow_html=True)

    vcol1, vcol2 = st.columns([1, 1])

    # ── LEFT: Mesh + Floorplates + Structure ──
    with vcol1:
        st.markdown('<div style="font-size:6px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.3; margin-bottom:5px;">VOLUME / FLOORPLATES / WIREFRAME</div>', unsafe_allow_html=True)
        mesh = load_building_mesh(uploaded_file=uploaded_mesh)
        mx, my, mz, mi, mj, mk = extract_plotly_mesh(mesh)

        fig1 = go.Figure()
        if len(mx) > 0:
            fig1.add_trace(go.Mesh3d(x=mx, y=my, z=mz, i=mi, j=mj, k=mk,
                opacity=0.04, color='rgba(100,160,255,0.15)', name='VOLUME', 
                lighting=dict(ambient=1, diffuse=0, specular=0), flatshading=True))

        z_levels, sqft, plate_paths = slice_mesh_and_get_floorplates(mesh, 0, num_floors*floor_height, floor_height)
        for p in plate_paths:
            fig1.add_trace(go.Scatter3d(x=p['x'], y=p['y'], z=p['z'], mode='lines',
                line=dict(color='rgba(255,255,255,0.12)', width=1.5), showlegend=False))

        edge_x, edge_y, edge_z = [], [], []
        for memb in last_members:
            n1 = last_model.nodes[memb['From']]
            n2 = last_model.nodes[memb['To']]
            edge_x.extend([n1.X, n2.X, None])
            edge_y.extend([n1.Y, n2.Y, None])
            edge_z.extend([n1.Z, n2.Z, None])
        fig1.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(color='rgba(255,255,255,0.6)', width=2), name='STRUCTURE'))

        node_x = [last_model.nodes[n].X for n in last_model.nodes]
        node_y = [last_model.nodes[n].Y for n in last_model.nodes]
        node_z = [last_model.nodes[n].Z for n in last_model.nodes]
        fig1.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
            marker=dict(size=1.5, color='rgba(255,255,255,0.4)'), name='NODES'))

        fig1.update_layout(**PLOT_LAYOUT)
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    # ── RIGHT: MIDAS displacement gradient ──
    with vcol2:
        st.markdown('<div style="font-size:6px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.3; margin-bottom:5px;">SEISMIC DISPLACEMENT GRADIENT</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        max_d = max(m['Max_Disp_in'], 0.001)

        for memb in last_members:
            n1 = last_model.nodes[memb['From']]
            n2 = last_model.nodes[memb['To']]
            d1 = node_disps.get(memb['From'], 0)
            d2 = node_disps.get(memb['To'], 0)
            fig2.add_trace(go.Scatter3d(
                x=[n1.X, n2.X], y=[n1.Y, n2.Y], z=[n1.Z, n2.Z], mode='lines',
                line=dict(width=4, color=[d1, d2], colorscale='Turbo', cmin=0, cmax=max_d),
                showlegend=False))

        # Colorbar reference
        fig2.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=0, colorscale='Turbo', cmin=0, cmax=max_d,
                colorbar=dict(title=dict(text='DRIFT (IN)', font=dict(size=7, family='Fragment Mono')),
                    thickness=8, len=0.6, tickfont=dict(size=7, family='Fragment Mono'), x=1.02)),
            showlegend=False))

        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── COMPARATIVE BAR CHARTS ──
    st.markdown('<hr style="border-color:rgba(255,255,255,0.04); margin: 20px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:7px; letter-spacing:0.4em; text-transform:uppercase; opacity:0.3; margin-bottom:10px; text-align:center;">COMPARATIVE OUTPUT</div>', unsafe_allow_html=True)

    df_metrics = pd.DataFrame([it['Metrics'] for it in st.session_state.iterations])
    BAR_LAYOUT = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, b=30, t=30), height=220,
        font=dict(family='Fragment Mono', size=7, color='rgba(255,255,255,0.5)'),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)')
    )

    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        fig_b1 = go.Figure(data=[go.Bar(x=df_metrics['Iteration'], y=df_metrics['Carbon_kgCO2e'],
            marker_color='rgba(100,200,100,0.6)', marker_line=dict(color='rgba(100,200,100,0.9)', width=1))])
        fig_b1.update_layout(title=dict(text='CARBON (KGCO2E)', font=dict(size=8)), **BAR_LAYOUT)
        st.plotly_chart(fig_b1, use_container_width=True)
    with bcol2:
        fig_b2 = go.Figure(data=[go.Bar(x=df_metrics['Iteration'], y=df_metrics['Cost_USD'],
            marker_color='rgba(100,150,255,0.6)', marker_line=dict(color='rgba(100,150,255,0.9)', width=1))])
        fig_b2.update_layout(title=dict(text='COST (USD)', font=dict(size=8)), **BAR_LAYOUT)
        st.plotly_chart(fig_b2, use_container_width=True)
    with bcol3:
        fig_b3 = go.Figure(data=[go.Bar(x=df_metrics['Iteration'], y=df_metrics['Volume_in3'],
            marker_color='rgba(255,160,60,0.6)', marker_line=dict(color='rgba(255,160,60,0.9)', width=1))])
        fig_b3.update_layout(title=dict(text='VOLUME (IN³)', font=dict(size=8)), **BAR_LAYOUT)
        st.plotly_chart(fig_b3, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; opacity:0.3;">
        <div style="font-size:8px; letter-spacing:0.4em; text-transform:uppercase;">EVALUATE A DESIGN TO UNLOCK VISUALIZERS</div>
    </div>
    """, unsafe_allow_html=True)
