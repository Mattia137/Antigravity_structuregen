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
        Ask API to increment structural profiles or add bracing to minimize carbon/cost.
        """
        from src.fea_solver import FEASolver
        
        print(f"Starting Evolutionary Optimization (Max Iterations: {max_iterations})")
        
        current_feedback = "Initial Design Generation. Focus on baseline stability and LABC compliance."
        latest_graph = None
        best_results = None
        
        for i in range(1, max_iterations + 1):
            print(f"\n[Iteration {i}/{max_iterations}] Generating AI Design...")
            
            # Append feedback to geometry data to inform the AI
            modified_geometry = dict(base_geometry)
            modified_geometry["optimization_feedback"] = current_feedback
            
            # 1. Ask API to mutate topology based on feedback
            design_json = self.ai.request_design(modified_geometry)
            struct_graph = self.ai.construct_graph(design_json)
            latest_graph = struct_graph
            
            # 2. Run FEA Solver
            fea = FEASolver(struct_graph, material_params)
            fea.build_model()
            fea.apply_loads()
            results = fea.solve_and_evaluate()
            best_results = results
            
            if results["status"] == "Passed":
                print(f"Design Passed FEA limits at iteration {i}!")
                break
            else:
                failures = results["failures"]
                print(f"Design Failed with {len(failures)} weak members.")
                
                # Compress failures to avoid token bloat
                fail_summary = failures[:10] 
                current_feedback = (
                    "FEA Analysis Failed on previous iteration.\n"
                    f"Max Nodal Displacement: {results.get('max_displacement', 0):.4f}m.\n"
                    f"Top Failing Members (Deflection > L/360): {fail_summary}\n"
                    "Mandatory Command: Mutate the topology. Increase specific member profiles (cross-sections) "
                    "or add localized cross-bracing targeting these weak topological zones to fix failures "
                    "while strictly minimizing the overall Carbon Footprint and Material Cost."
                )

        print("Optimization Pipeline Finished.")
        return latest_graph, best_results

if __name__ == "__main__":
    print("Evolutionary Optimizer Ready.")
