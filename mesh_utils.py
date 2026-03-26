import trimesh
import numpy as np

def parse_rhino_obj_lines(filepath):
    """
    Parses a Rhino exported OBJ file to extract degree-1 bsplines (polylines)
    and their associated 'usemtl' colors, treating Red(255,0,0) as Primary
    and Blue(0,x,255) as Secondary.
    """
    vertices = []
    lines = [] # list of (role, [list of vertex indices])
    
    current_role = 'Secondary'
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    
                elif line.startswith('usemtl '):
                    matname = line.split()[1].lower()
                    if '255_0_0' in matname:
                        current_role = 'Primary'
                    elif '0_29_255' in matname or '0_0_255' in matname:
                        current_role = 'Secondary'
                    else:
                        current_role = 'Secondary'
                        
                elif line.startswith('curv '):
                    parts = line.split()
                    idx_list = []
                    for p in parts[3:]:
                        try:
                            idx_list.append(int(p) - 1)  # obj is 1-indexed
                        except ValueError:
                            pass
                    if idx_list:
                        lines.append((current_role, idx_list))
    except Exception as e:
        print(f"Error parsing Rhino obj lines: {e}")

    return vertices, lines

def load_building_mesh(uploaded_file=None, filepath="structure-example.obj"):
    """
    Loads mesh AND extracts raw structural polylines for wireframe-first generation.
    Returns: (trimesh_object, vertices_list, lines_list)
    """
    vertices, lines = [], []
    mesh = None
    
    try:
        if uploaded_file is not None:
            import io
            ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
            mesh = trimesh.load(io.BytesIO(uploaded_file.read()), file_type=ext, force='mesh')
            # Extract lines from uploaded lines if any
            pass
        else:
            mesh = trimesh.load(filepath, force='mesh')
            vertices, lines = parse_rhino_obj_lines(filepath)
            
        # Task 1: If it's a closed mesh with no explicit lines, generate a triangulated diagrid
        if len(lines) == 0 and isinstance(mesh, trimesh.Trimesh):
            print("No explicitly defined wireframes found. Extracting Triangulated Diagrid from Mesh Surface...")
            vertices = [tuple(v) for v in mesh.vertices]
            
            # Find creases (dihedral angle > 20 degrees)
            try:
                adjacency_angles = np.abs(mesh.face_adjacency_angles)
                is_crease = adjacency_angles > np.radians(20)
                crease_edges = mesh.face_adjacency_edges[is_crease]
                
                # Primary Load Path (Creases)
                crease_set = set()
                for edge in crease_edges:
                    crease_tuple = tuple(sorted(edge))
                    crease_set.add(crease_tuple)
                    lines.append(('Primary', list(edge)))
                    
                # Secondary Lattice (Remaining Delaunay Triangulation edges)
                secondary_edges = set()
                for edge in mesh.edges_unique:
                    edge_tuple = tuple(sorted(edge))
                    if edge_tuple not in crease_set:
                        secondary_edges.add(edge_tuple)
                
                for edge in secondary_edges:
                    lines.append(('Secondary', list(edge)))
            except Exception as e:
                print(f"Diagrid generation failed: {e}")
                # Fallback to all edges as secondary
                for edge in mesh.edges_unique:
                    lines.append(('Secondary', list(edge)))

    except Exception as e:
        print(f"Warning: Could not load mesh. ({e})")
        box = trimesh.creation.box(extents=[720, 720, 1440])
        box.apply_translation([0, 0, 720])
        mesh = box

    return mesh, vertices, lines

def extract_plotly_mesh(mesh):
    """Extracts raw vertex and face data for Plotly's Mesh3d visualizer."""
    if not isinstance(mesh, trimesh.Trimesh):
        return [], [], [], [], [], []
    
    vertices = mesh.vertices
    faces = mesh.faces
    
    x = vertices[:, 0].tolist()
    y = vertices[:, 1].tolist()
    z = vertices[:, 2].tolist()
    
    i = faces[:, 0].tolist()
    j = faces[:, 1].tolist()
    k = faces[:, 2].tolist()
    
    return x, y, z, i, j, k

def slice_mesh_and_get_floorplates(mesh, z_start, z_end, step):
    """
    Slices the mesh at every Z elevation to emulate a floor-by-floor generator.
    Returns:
      z_levels: list of elevations
      total_sqft: computed total gross floor area
      plate_paths: List of coordinates dicts {x, y, z} to draw boundary contours in 3D
    """
    z_levels = np.arange(z_start, z_end, step)
    total_sqft = 0.0
    plate_paths = []
    
    # Pre-scale bounds to avoid massive computation times if mesh is crazy
    try:
        if isinstance(mesh, trimesh.Trimesh):
            for z in z_levels:
                slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if slice_3d is not None:
                    # Sum area via Planar projection
                    slice_2d, _ = slice_3d.to_planar()
                    floor_area = sum([p.area for p in slice_2d.polygons_full])
                    total_sqft += (floor_area / 144.0)
                    
                    # Extract 3D polygonal boundaries for drawing
                    for entity in slice_3d.entities:
                        # entities.points points to an index in slice_3d.vertices
                        pts = slice_3d.vertices[entity.points]
                        px = pts[:, 0].tolist() + [pts[0, 0]]
                        py = pts[:, 1].tolist() + [pts[0, 1]]
                        pz = pts[:, 2].tolist() + [pts[0, 2]]
                        plate_paths.append({'x': px, 'y': py, 'z': pz})
    except Exception as e:
        # Fallback GFA
        vol = mesh.bounding_box.volume
        h = mesh.bounding_box.extents[2]
        area_sqin = vol / h if h > 0 else 0
        total_sqft = (area_sqin / 144.0) * len(z_levels)
        
    return z_levels, total_sqft, plate_paths
