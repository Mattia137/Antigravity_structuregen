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
                    
                    floors.append({
                        "elevation_z": z,
                        "area_sqft": area_sqft
                    })
            except Exception as e:
                pass
                
        return {
            "num_floors": len(floors),
            "total_sqft": total_sqft,
            "floors": floors
        }

if __name__ == "__main__":
    # Example usage / test stub
    print("Geometry Engine Ready.")
