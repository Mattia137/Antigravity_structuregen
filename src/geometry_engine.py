import numpy as np
import trimesh

class GeometryEngine:
    def __init__(self, filepath: str):
        """
        Ingest .obj/.stl meshes, center them at (0,0,0) and align the base to Z=0.
        """
        # process=False ensures raw coordinates are preserved exactly
        self.mesh = trimesh.load_mesh(filepath, process=False)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.dump(concatenate=True)
            
        # Manually perform unique vertex identification ONLY if needed, without averaging
        # self.mesh.merge_vertices() # Removd to avoid potential coordinate shifts
        self.normalize_mesh()

        print(f"Loaded and normalized mesh {filepath}: {len(self.mesh.vertices)} vertices.")

    def normalize_mesh(self):
        """
        Center the mesh in X and Y, and move the lowest point to Z=0.
        """
        extents = self.mesh.bounds
        center_xy = (extents[0][:2] + extents[1][:2]) / 2.0
        z_min = extents[0][2]
        
        # Translation vector: offset to center XY and bring Z_min to 0
        translation = np.array([-center_xy[0], -center_xy[1], -z_min])
        self.mesh.apply_translation(translation)

    def extract_boundary_nodes(self):
        """
        Extract all vertices as boundary nodes.
        Returns the raw (X, Y, Z) coordinates.
        """
        return self.mesh.vertices.tolist()

    def extract_primary_creases(self, angle_threshold_degrees=20.0):
        """
        Perform dihedral angle analysis to isolate "creases."
        These creases represent the foundational lines for the primary structural frame.
        Returns the edges and the unique vertex indices forming those creases.
        """
        if not hasattr(self.mesh, 'face_adjacency_angles'):
            # In case mesh does not have adjacency computed
            return {"edges": [], "nodes": []}

        threshold_rad = np.radians(angle_threshold_degrees)
        
        angles = self.mesh.face_adjacency_angles
        edges = self.mesh.face_adjacency_edges
        
        # A crease exists where the dihedral angle > threshold
        crease_mask = np.abs(angles) > threshold_rad
        crease_edges = edges[crease_mask]
        
        # Unique node indices for the creases
        crease_nodes = np.unique(crease_edges.flatten())
        
        return {
            "edges": crease_edges.tolist(),
            "nodes": crease_nodes.tolist()
        }

    def slice_mesh_horizontally(self, floor_height=4.0):
        """
        Slice the mesh horizontally to determine floor heights and calculate total usable SqFt.
        Returns total square footage and per-floor data.
        """
        bounds = self.mesh.bounds
        z_min, z_max = bounds[0][2], bounds[1][2]
        
        total_height = z_max - z_min
        num_floors = int(np.floor(total_height / floor_height))
        
        if num_floors == 0:
            num_floors = 1

        z_levels = [z_min + (i * floor_height) for i in range(1, num_floors + 1)]
        
        total_sqft = 0.0
        floors = []
        
        for z in z_levels:
            try:
                cross_section = self.mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if cross_section is not None:
                    planar, _ = cross_section.to_planar()
                    area_m2 = planar.area
                    area_sqft = area_m2 * 10.7639
                    total_sqft += area_sqft
                    
                    centroid_3d = cross_section.centroid
                    
                    # 200 ft rule: 60.96m. If bounding box diag > 2 * 60.96, multiple cores needed.
                    max_dist = np.linalg.norm(cross_section.bounds[1][:2] - cross_section.bounds[0][:2]) / 2.0
                    suggested_core_count = int(np.ceil(max_dist / 60.96))
                    
                    floors.append({
                        "elevation_z": z,
                        "area_sqft": area_sqft,
                        "centroid": [float(centroid_3d[0]), float(centroid_3d[1])],
                        "suggested_core_count": max(1, suggested_core_count),
                        "bounds_2d": {
                            "x_min": float(cross_section.bounds[0][0]),
                            "x_max": float(cross_section.bounds[1][0]),
                            "y_min": float(cross_section.bounds[0][1]),
                            "y_max": float(cross_section.bounds[1][1])
                        }
                    })
            except Exception as e:
                pass
                
        return {
            "num_floors": len(floors),
            "total_sqft": total_sqft,
            "floors": floors,
            "avg_floor_area": total_sqft / len(floors) if floors else 0
        }

    def get_max_height_points(self):
        """
        Identify the (X, Y) coordinates where the building reaches its maximum Z.
        This is where elevator cores MUST be placed to reach the rooftop.
        Additionally, enforce the 200ft (60.96m) rule across the entire floorplate
        so that NO point on ANY horizontal floor plate is more than 200ft from a core.
        """
        verts = self.mesh.vertices
        if len(verts) == 0:
            return [[0.0, 0.0]]

        max_z = np.max(verts[:, 2])

        # 1. Base Cores: Must be at maximum height to reach the roof
        peak_mask = verts[:, 2] > (max_z - 0.5)
        peak_verts = verts[peak_mask]
        
        clusters = []
        for v in peak_verts:
            pt = np.array([v[0], v[1]])
            added = False
            for c in clusters:
                if np.linalg.norm(c["center"] - pt) < 60.96:
                    c["points"].append(pt)
                    c["center"] = np.mean(c["points"], axis=0)
                    added = True
                    break
            if not added:
                clusters.append({"points": [pt], "center": pt})

        core_centers = [c["center"] for c in clusters]

        # 2. 200ft Distance Rule Enforcement
        # Check all vertices. If any vertex is > 60.96m from ALL existing cores,
        # we must add a new core. (We'll use a crude clustering for missing coverage)

        uncovered_pts = []
        for v in verts:
            pt = np.array([v[0], v[1]])
            # Compute distance to all cores
            min_dist = min([np.linalg.norm(core - pt) for core in core_centers]) if core_centers else float('inf')

            if min_dist > 60.96:
                uncovered_pts.append(pt)

        if uncovered_pts:
            # Cluster the uncovered points to form new cores
            new_clusters = []
            for pt in uncovered_pts:
                added = False
                for c in new_clusters:
                    if np.linalg.norm(c["center"] - pt) < 60.96:
                        c["points"].append(pt)
                        c["center"] = np.mean(c["points"], axis=0)
                        added = True
                        break
                if not added:
                    new_clusters.append({"points": [pt], "center": pt})

            for c in new_clusters:
                core_centers.append(c["center"])

        return [[float(center[0]), float(center[1])] for center in core_centers]

    @staticmethod
    def generate_solid_structure(graph, node_displacements=None):
        """
        Generate a solid 3D mesh (trimesh.Trimesh) by extruding cylinders 
        along each graph edge. Radius is derived from section area.
        Also attaches vertex colors based on displacement if provided.
        """
        from config import SECTIONS
        import networkx as nx
        
        all_meshes = []
        
        # Determine max displacement for normalization
        max_disp = 0
        if node_displacements:
            max_disp = max(node_displacements.values()) if node_displacements.values() else 0

        def get_color(disp, max_d):
            if max_d <= 0: return [255, 255, 255, 255]
            ratio = disp / max_d
            # Blue (0) to Red (1)
            r = int(ratio * 255)
            b = int((1 - ratio) * 255)
            return [r, 0, b, 255]

        for u, v, data in graph.edges(data=True):
            p1 = np.array(graph.nodes[u]["coords"])
            p2 = np.array(graph.nodes[v]["coords"])
            
            section_name = data.get("section", "IPE_300")
            area_in2 = 8.34 # default
            for mat_type in SECTIONS:
                if section_name in SECTIONS[mat_type]:
                    area_in2 = SECTIONS[mat_type][section_name]["A"]
                    break
            
            area_m2 = area_in2 * 6.4516e-4
            radius = np.sqrt(area_m2 / np.pi) * 2.0 # Scale up slightly for visibility
            
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            if length < 1e-6: continue
            
            cylinder = trimesh.creation.cylinder(radius=radius, height=length)
            
            z_axis = [0, 0, 1]
            rotation = trimesh.geometry.align_vectors(z_axis, edge_vec)
            
            midpoint = (p1 + p2) / 2.0
            matrix = np.eye(4)
            matrix[:3, :3] = rotation[:3, :3]
            matrix[:3, 3] = midpoint
            
            cylinder.apply_transform(matrix)

            # Apply colors based on displacement of nodes
            if node_displacements:
                d_u = node_displacements.get(str(u), 0.0)
                d_v = node_displacements.get(str(v), 0.0)
                d_avg = (d_u + d_v) / 2.0
                color = get_color(d_avg, max_disp)
                cylinder.visual.vertex_colors = [color for _ in cylinder.vertices]

            all_meshes.append(cylinder)
            
        if not all_meshes:
            return None
        return trimesh.util.concatenate(all_meshes)

if __name__ == "__main__":
    # Example usage / test stub
    print("Geometry Engine Ready.")
