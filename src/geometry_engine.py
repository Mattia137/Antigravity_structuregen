import numpy as np
import trimesh

class GeometryEngine:
    def __init__(self, filepath: str):
        """
        Ingest .obj/.stl meshes with absolute 1:1 scale fidelity.
        Never normalize or center coordinates.
        """
        # process=False ensures coordinates, normals, etc are not structurally altered/normalized
        self.mesh = trimesh.load_mesh(filepath, process=False)
        if isinstance(self.mesh, trimesh.Scene):
            self.mesh = self.mesh.dump(concatenate=True)
            
        print(f"Loaded mesh {filepath} with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces.")

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

    def slice_mesh_horizontally(self, floor_height=3.0):
        """
        Slice the mesh horizontally to determine floor heights and calculate total usable SqFt.
        Assumes Z is the vertical axis and units are meters.
        Returns total square footage and per-floor data.
        """
        bounds = self.mesh.bounds
        z_min, z_max = bounds[0][2], bounds[1][2]
        
        total_height = z_max - z_min
        num_floors = int(np.floor(total_height / floor_height))
        
        z_levels = [z_min + (i * floor_height) for i in range(1, num_floors + 1)]
        
        total_sqft = 0.0
        floors = []
        
        for z in z_levels:
            try:
                # Get cross section at elevation z
                cross_section = self.mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if cross_section is not None:
                    # Convert to planar 2D to calculate area safely
                    planar, _ = cross_section.to_planar()
                    area_m2 = planar.area
                    area_sqft = area_m2 * 10.7639
                    total_sqft += area_sqft
                    
                    # Calculate centroids for core positioning advice
                    centroid_3d = cross_section.centroid
                    
                    # Core Coverage Logic (60m / 200ft rule)
                    # For a crude approximation, check the 'radius' of the floor
                    # If the distance from centroid to furthest boundary point is > 60m, suggest multiple cores
                    max_dist = np.linalg.norm(cross_section.bounds[1][:2] - cross_section.bounds[0][:2]) / 2
                    suggested_core_count = int(np.ceil(max_dist / 60.0))
                    
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
        """
        verts = self.mesh.vertices
        max_z = np.max(verts[:, 2])
        # Find all vertices at roughly the max height (1m tolerance)
        peak_mask = verts[:, 2] > (max_z - 1.0)
        peak_verts = verts[peak_mask]
        
        # Return unique X, Y coordinates for these peak points
        # Clustering would be better, but for now take the average of local maxima
        return [[float(np.mean(peak_verts[:, 0])), float(np.mean(peak_verts[:, 1]))]]

    def sample_internal_nodes(self, grid_spacing=5.0):
        """
        Generate a 3D grid of points within the mesh volume.
        Used to provide Gemini with anchor points for internal structural networks.
        """
        bounds = self.mesh.bounds
        x_range = np.arange(bounds[0][0], bounds[1][0], grid_spacing)
        y_range = np.arange(bounds[0][1], bounds[1][1], grid_spacing)
        z_range = np.arange(bounds[0][2], bounds[1][2], grid_spacing)
        
        # Create 3D grid
        xv, yv, zv = np.meshgrid(x_range, y_range, z_range)
        points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
        
        # Filter points that are actually inside the mesh
        inside_mask = self.mesh.contains(points)
        internal_points = points[inside_mask]
        
        print(f"Sampled {len(internal_points)} internal nodes within mesh volume.")
        return internal_points.tolist()

    def generate_solid_structure(self, graph):
        """
        Generate a solid 3D mesh (trimesh.Trimesh) by extruding cylinders 
        along each graph edge. Radius is derived from section area.
        """
        from config import SECTIONS
        import networkx as nx
        
        all_meshes = []
        
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
            radius = np.sqrt(area_m2 / np.pi)
            
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            if length < 1e-6: continue
            
            cylinder = trimesh.creation.cylinder(radius=radius, height=length)
            
            z_axis = [0, 0, 1]
            rotation, _ = trimesh.geometry.align_vectors(z_axis, edge_vec)
            
            midpoint = (p1 + p2) / 2.0
            matrix = np.eye(4)
            matrix[:3, :3] = rotation[:3, :3]
            matrix[:3, 3] = midpoint
            
            cylinder.apply_transform(matrix)
            all_meshes.append(cylinder)
            
        if not all_meshes:
            return None
        return trimesh.util.concatenate(all_meshes)

if __name__ == "__main__":
    # Example usage / test stub
    print("Geometry Engine Ready.")
