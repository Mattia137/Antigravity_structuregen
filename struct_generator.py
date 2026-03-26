from Pynite import FEModel3D
import config
import trimesh

def generate_structure(vertices, lines, num_floors, floor_height, bay_size=240, selected_material='Steel', custom_props=None, member_overrides=None):
    model = FEModel3D()
    
    props = custom_props if custom_props else config.MATERIALS[selected_material]
    E = props['E']
    nu = props['nu']
    G = E / (2*(1+nu))
    rho = props['rho']
    
    model.add_material(selected_material, E, G, nu, rho)
    
    # Load all standard Sections into PyNite
    sections_dict = config.SECTIONS.get(selected_material, {})
    for sec_name, s_props in sections_dict.items():
        model.add_section(sec_name, s_props['A'], s_props['Iy'], s_props['Iz'], s_props['J'])
        
    if not vertices:
        return model, [], {}

    # Find the extremes of the provided wireframe structure
    valid_z = [v[2] for v in vertices]
    min_z = min(valid_z)
    max_z = max(valid_z)
    
    # Floor elevations from base to max allowed by num_floors
    floor_zs = [min_z + i * floor_height for i in range(int(num_floors)+1)]
    
    def intersect_z(p1, p2, z):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        if abs(z2 - z1) < 1e-6: return None
        t = (z - z1) / (z2 - z1)
        if 0 < t < 1:
            return (x1 + t*(x2-x1), y1 + t*(y2-y1), z)
        return None
        
    unsplit_segments = []
    for role, indices in lines:
        for i in range(len(indices)-1):
            p1 = vertices[indices[i]]
            p2 = vertices[indices[i+1]]
            if p1 != p2:
                unsplit_segments.append((role, p1, p2))
                
    final_segments = []
    for role, p1, p2 in unsplit_segments:
        ints = [p1, p2]
        for fz in floor_zs:
            p_int = intersect_z(p1, p2, fz)
            if p_int:
                ints.append(p_int)
        
        def dist(pa, pb):
            return sum((a-b)**2 for a, b in zip(pa, pb))**0.5
            
        ints.sort(key=lambda p: dist(p1, p))
        
        # Clean duplicates
        clean_ints = [ints[0]]
        for p in ints[1:]:
            if dist(clean_ints[-1], p) > 1.0: # 1 inch tolerance
                clean_ints.append(p)
                
        for i in range(len(clean_ints)-1):
            final_segments.append((role, clean_ints[i], clean_ints[i+1]))

    # GRAPH CONNECTIVITY CHECK (Remove Floating Nodes)
    adjacency = {}
    for role, pt1, pt2 in final_segments:
        if pt1 not in adjacency: adjacency[pt1] = []
        if pt2 not in adjacency: adjacency[pt2] = []
        adjacency[pt1].append(pt2)
        adjacency[pt2].append(pt1)
        
    foundation_nodes = [pt for pt in adjacency.keys() if abs(pt[2] - min_z) < 1.0]
    
    # BFS to find all connected components tied to foundation
    connected_nodes = set(foundation_nodes)
    queue = list(foundation_nodes)
    
    while queue:
        curr = queue.pop(0)
        for neighbor in adjacency[curr]:
            if neighbor not in connected_nodes:
                connected_nodes.add(neighbor)
                queue.append(neighbor)
                
    # Filter final_segments to only those whose both nodes are connected
    valid_segments = []
    for role, pt1, pt2 in final_segments:
        if pt1 in connected_nodes and pt2 in connected_nodes:
            valid_segments.append((role, pt1, pt2))
            
    final_segments = valid_segments

    nodes_xyz = {}
    node_id = 0
    def get_node(pt):
        nonlocal node_id
        key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
        if key not in nodes_xyz:
            name = f'N_{node_id}'
            node_id += 1
            model.add_node(name, pt[0], pt[1], pt[2])
            nodes_xyz[key] = name
            
            # Ground support (allow slight tolerance)
            if abs(pt[2] - min_z) < 1.0:
                model.def_support(name, True, True, True, True, True, True)
                
        return nodes_xyz[key]

    members = []
    mem_id = 0
    defaults = config.DEFAULTS.get(selected_material, {'Primary':'?', 'Secondary':'?', 'Diagonal':'?'})

    for role, pt1, pt2 in final_segments:
        n1 = get_node(pt1)
        n2 = get_node(pt2)
        mname = f'M_{mem_id}'
        mem_id += 1
        
        assigned_sec = defaults.get(role, defaults.get('Primary', 'W14x90'))
        if member_overrides and mname in member_overrides:
            assigned_sec = member_overrides[mname]
            
        model.add_member(mname, n1, n2, selected_material, assigned_sec)
        
        # Pin the exterior diagrid nodes (axial only), keep Core/Floor Fixed.
        if role in ['Primary', 'Secondary']:
            model.def_releases(mname, Dxi=False, Dyi=False, Dzi=False, Rxi=False, Ryi=True, Rzi=True,
                                      Dxj=False, Dyj=False, Dzj=False, Rxj=False, Ryj=True, Rzj=True)
                                      
        members.append({
            'MemberID': mname,
            'From': n1,
            'To': n2,
            'StructuralRole': role,
            'SectionType': assigned_sec
        })

    # CONCRETE CORE & FLOOR PLATES
    all_x = [pt[0] for pt in vertices]
    all_y = [pt[1] for pt in vertices]
    cx = (max(all_x) + min(all_x)) / 2 if all_x else 0
    cy = (max(all_y) + min(all_y)) / 2 if all_y else 0
    
    core_nodes = {}
    core_sec = 'Core_Massive'

    for fz in floor_zs:
        if fz > max_z + 1.0: break # Keep core inside physical height bounds
        
        c_node = get_node((cx, cy, fz))
        core_nodes[fz] = c_node
        
        idx = floor_zs.index(fz)
        if idx > 0:
            prev_fz = floor_zs[idx-1]
            if prev_fz in core_nodes:
                c1 = core_nodes[prev_fz]
                c2 = c_node
                mname = f'Core_{mem_id}'
                mem_id += 1
                model.add_member(mname, c1, c2, selected_material, core_sec)
                members.append({'MemberID': mname, 'From': c1, 'To': c2, 'StructuralRole': 'Core', 'SectionType': core_sec})
        
        # Radial Floor Diaphragm / Tie beams
        floor_nodes = [name for k, name in nodes_xyz.items() if abs(k[2] - fz) < 1.0 and name != c_node]
        
        for fn in floor_nodes:
            mname = f'FloorTie_{mem_id}'
            mem_id += 1
            sec = defaults.get('Secondary', 'W12x50')
            model.add_member(mname, fn, c_node, selected_material, sec)
            members.append({'MemberID': mname, 'From': fn, 'To': c_node, 'StructuralRole': 'Floor', 'SectionType': sec})

    # LOADING
    base_shear = config.SEISMIC['S_DS'] * 10 * bay_size * floor_height * 0.001
    model.add_load_combo('Seismic', {'D': 1.0, 'E': 1.0})
    
    for fz in floor_zs:
        if fz <= min_z + 1.0: continue
        
        floor_keys = [k for k in nodes_xyz.keys() if abs(k[2] - fz) < 1.0]
        if not floor_keys: continue
        
        # Gravity
        for k in floor_keys:
            model.add_node_load(nodes_xyz[k], 'FZ', -1, case='D')
            
        # Lateral pushes on Core Nodes representing diaphragm shear
        dist_lat = base_shear * (fz / max_z)
        model.add_node_load(c_node, 'FX', dist_lat, case='E')

    return model, members, nodes_xyz
