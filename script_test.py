import sys

def parse_rhino_obj_lines(filepath):
    vertices = []
    lines = [] # list of (color, [list of vertex indices])
    
    current_color = 'default'
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                
            elif line.startswith('usemtl '):
                # e.g., usemtl diffuse_255_0_0_255
                matname = line.split()[1]
                if '255_0_0' in matname:
                    current_color = 'red'
                elif '0_29_255' in matname or '0_0_255' in matname:
                    current_color = 'blue'
                else:
                    current_color = 'default'
                    
            elif line.startswith('curv '):
                # e.g., curv 0 98.20530923610562 1 2 3 4 5 6 7 8 9 10 11
                parts = line.split()
                # indices start at index 3 in the parts list
                idx_list = []
                for p in parts[3:]:
                    try:
                        idx_list.append(int(p) - 1)  # obj is 1-indexed
                    except ValueError:
                        pass
                if idx_list:
                    lines.append((current_color, idx_list))

    return vertices, lines

if __name__ == "__main__":
    v, l = parse_rhino_obj_lines('structure-example.obj')
    print(f"Parsed {len(v)} vertices.")
    print(f"Parsed {len(l)} lines.")
    
    red_lines = [line for color, line in l if color == 'red']
    blue_lines = [line for color, line in l if color == 'blue']
    
    print(f"Red lines: {len(red_lines)}")
    print(f"Blue lines: {len(blue_lines)}")
    
    z_coords = [round(v_z, 2) for _, _, v_z in v]
    unique_z = sorted(list(set(z_coords)))
    print(f"Unique Z heights ({len(unique_z)}):")
    print(unique_z[:20])
    print(f"Min Z: {min(z_coords)}, Max Z: {max(z_coords)}")
