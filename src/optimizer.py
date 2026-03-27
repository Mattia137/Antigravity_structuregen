import networkx as nx

class EvolutionaryOptimizer:
    def __init__(self, ai_designer):
        """
        Initializes the optimization engine with an instance of the AI Designer.
        """
        self.ai = ai_designer

    def run_optimization_loop(self, base_geometry: dict, material_params: dict, max_iterations=3):
        """
        Executes the feedback loop sending FEA failing zones back to the AI.
        Outputs exactly 3 optimized versions: Lowest Cost, Lowest Carbon Footprint, Balanced.
        """
        from src.fea_solver import FEASolver
        
        print(f"Starting Evolutionary Optimization (Max Iterations: {max_iterations})")
        
        # We need to return exactly 3 optimized variants.
        goals = ["DISPLACEMENT", "COST", "CARBON"]
        final_variants = []
        
        for goal in goals:
            current_feedback = "Initial Design Generation. Focus on baseline stability and LABC compliance."
            latest_graph = None
            best_results = None
            latest_design_json = None
            
            for i in range(1, max_iterations + 1):
                print(f"\n[Iteration {i}/{max_iterations}] Generating AI Design for {goal}...")

                modified_geometry = dict(base_geometry)
                modified_geometry["optimization_feedback"] = current_feedback

                design_json = self.ai.request_design(modified_geometry, optimization_goal=goal)
                struct_graph = self.ai.construct_graph(design_json)
                latest_graph = struct_graph
                latest_design_json = design_json

                fea = FEASolver(struct_graph, material_params)
                fea.build_model()
                fea.apply_loads()
                results = fea.solve_and_evaluate()
                best_results = results

                if results["status"] == "Passed":
                    print(f"Design Passed FEA limits at iteration {i} for {goal}!")
                    break
                else:
                    failures = results["failures"]
                    print(f"Design Failed with {len(failures)} weak members.")

                    fail_summary = failures[:10]
                    current_feedback = (
                        "FEA Analysis Failed on previous iteration.\n"
                        f"Max Nodal Displacement: {results.get('max_displacement', 0):.4f}m.\n"
                        f"Top Failing Members (Deflection > L/360): {fail_summary}\n"
                        "Mandatory Command: Mutate the topology. Increase specific member profiles (cross-sections) "
                        "or add localized cross-bracing targeting these weak topological zones to fix failures "
                        "while strictly minimizing the overall Carbon Footprint and Material Cost."
                    )
            
            # Compute heuristic metrics for the dashboard
            total_length = 0.0
            for u, v, data in latest_graph.edges(data=True):
                import numpy as np
                coord_u = np.array(latest_graph.nodes[u]["coords"])
                coord_v = np.array(latest_graph.nodes[v]["coords"])
                total_length += np.linalg.norm(coord_u - coord_v)

            total_volume = total_length * 0.05
            rho = material_params.get("rho", 7850)
            total_mass = total_volume * rho
            mat_type = material_params.get("type", "Steel")
            
            if mat_type == "Steel":
                total_carbon = total_mass * 1.22
                total_cost = (total_mass / 1000) * 2653.0
            else:
                total_carbon = total_mass * 0.20
                total_cost = (total_mass / 1833) * 145.0

            # Build node and member structure expected by UI variants
            active_nodes = {}
            for n, data in latest_graph.nodes(data=True):
                coords = data["coords"]
                active_nodes[str(n)] = {
                    "x": coords[0],
                    "y": coords[1],
                    "z": coords[2],
                    "connection_type": data.get("connection_type", "welded")
                }

            active_members = []
            for u, v, m_data in latest_graph.edges(data=True):
                disp_i = best_results["node_displacements"].get(str(u), 0.0) if best_results and "node_displacements" in best_results else 0.0
                disp_j = best_results["node_displacements"].get(str(v), 0.0) if best_results and "node_displacements" in best_results else 0.0
                active_members.append({
                    "from": str(u),
                    "to": str(v),
                    "disp_i": disp_i,
                    "disp_j": disp_j,
                    "section": m_data.get("section", "unknown"),
                    "typology": m_data.get("typology", "unknown")
                })

            final_variants.append({
                "name": goal,
                "graph": latest_graph,
                "design_json": latest_design_json,
                "best_results": best_results,
                "nodes": active_nodes,
                "members": active_members,
                "metrics": {
                    "Volume": total_volume,
                    "Mass": total_mass,
                    "Carbon_kgCO2e": total_carbon,
                    "Cost_USD": total_cost,
                    "Max_Disp": best_results.get("max_displacement", 0) if best_results else 0
                }
            })

        print("Optimization Pipeline Finished.")
        # Return the DISPLACEMENT variant as the default graph and results, but attach all variants to the graph object
        # Alternatively, since app.py expects a graph and best_results, we can pass the final_variants list via a custom attribute
        default_variant = next((v for v in final_variants if v["name"] == "DISPLACEMENT"), final_variants[0])
        default_variant["graph"].graph["variants"] = final_variants

        return default_variant["graph"], default_variant["best_results"]

if __name__ == "__main__":
    print("Evolutionary Optimizer Ready.")
