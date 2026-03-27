"""
Evolutionary Optimizer
======================
Pipeline:
  1. AI generates one BALANCED base design (single Gemini call)
  2. PyNite FEA runs on base design
  3. Structural rules code checks (ASCE 7-22 drift, IBC L/360, AISC KL/r≤200)
  4. If code checks FAIL and iterations remain → targeted revision feedback to Gemini
  5. After convergence → produce 3 variants via section scaling:
       MIN_COST  (-2 ladder steps)
       BALANCED  (as-is)
       MIN_DISP  (+2 ladder steps)
  6. Run FEA on all 3 variants; return list of {graph, fea, goal}
"""
import networkx as nx

# Section ladder: ordered lightest → heaviest by cross-sectional area
_STEEL_LADDER = [
    "W8x31", "IPE_300", "HEA_200", "HSS6x0.500",
    "W12x50", "W12x53", "Tubular_HSS_4x4x1/4",
    "HSS8x8x0.500", "HSS10x0.500",
    "W14x90", "W18x97", "W24x146",
    "W14x159", "HSS12x12x0.625", "HSS16x0.625", "W14x283"
]
_LADDER_INDEX = {s: i for i, s in enumerate(_STEEL_LADDER)}


def _scale_sections(graph, steps: int) -> nx.Graph:
    """Return a copy of graph with every steel section shifted ±steps on the ladder."""
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        sec = data.get("section", "")
        if sec in _LADDER_INDEX:
            new_idx = max(0, min(len(_STEEL_LADDER) - 1, _LADDER_INDEX[sec] + steps))
            G[u][v]["section"] = _STEEL_LADDER[new_idx]
    return G


def _run_fea(graph, material_params: dict) -> dict:
    from src.fea_solver import FEASolver
    fea = FEASolver(graph, material_params)
    fea.build_model()
    fea.apply_loads()
    return fea.solve_and_evaluate()


class EvolutionaryOptimizer:
    def __init__(self, ai_designer):
        self.ai = ai_designer

    # ------------------------------------------------------------------
    def run_optimization_loop(
        self,
        base_geometry: dict,
        material_params: dict,
        max_iterations: int = 1,
    ) -> tuple:
        """
        Run the generative + FEA + code-check revision pipeline.

        Returns
        -------
        (base_graph, base_fea_results)
        The calleruses build_three_variants() to produce cost/carbon/disp variants.
        """
        from src.structural_rules_bridge import run_code_checks

        mesh_desc = base_geometry.get("mesh_desc", {})
        feedback  = "Initial design. Ensure code compliance per ASCE 7-22 and AISC 360-22."
        best_graph   = None
        best_results = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n[Iter {iteration}/{max_iterations}] Requesting AI base design...")

            geom = dict(base_geometry)
            geom["optimization_feedback"] = feedback

            # ── 1. Gemini generates design ─────────────────────────────────
            design_json = self.ai.request_base_design(geom)
            graph       = self.ai.construct_graph(design_json)
            best_graph  = graph

            # ── 2. PyNite FEA ──────────────────────────────────────────────
            fea_results  = _run_fea(graph, material_params)
            best_results = fea_results

            # ── 3. Code checks ─────────────────────────────────────────────
            check_report = run_code_checks(fea_results, mesh_desc, graph)
            drift_DCR    = check_report.get("drift_DCR", 0.0)
            max_disp     = check_report.get("max_disp_m", 0.0)
            overall      = check_report.get("overall", "unknown")
            h_sx         = check_report.get("h_sx_m", 4.0)

            print(f"  FEA: δ_max={max_disp:.4f} m | drift_DCR={drift_DCR:.3f} | checks={overall}")

            if overall == "pass":
                print(f"  ✓ All code checks PASS at iteration {iteration}.")
                break

            if iteration >= max_iterations:
                print(f"  ✗ Code checks did not converge in {max_iterations} iterations.")
                break

            # ── 4. Build targeted feedback for next Gemini call ────────────
            fea_fails = fea_results.get("failures", [])[:8]
            check_fails = [c for c in check_report.get("checks", []) if c["status"] == "fail"]
            drift_fail  = any("drift" in c.get("type", "") for c in check_fails)
            slender_fail = [c for c in check_fails if "slenderness" in c.get("type", "")]

            parts = [
                f"ITERATION {iteration} REVISION REQUIRED:",
                f"  Max displacement = {max_disp:.4f} m (story height = {h_sx:.1f} m)",
                f"  Drift DCR = {drift_DCR:.3f} (limit 1.0 = ASCE 7-22 §12.12-1)",
            ]
            if drift_fail:
                parts.append(
                    "  DRIFT FAILS: Add X-bracing diagonals in ALL perimeter bays. "
                    "Upgrade ALL column sections by 2 ladder steps. "
                    "Add secondary bracing nodes at mid-height of columns > 6 m."
                )
            if fea_fails:
                member_ids = [f["member"] for f in fea_fails]
                parts.append(
                    f"  L/360 DEFLECTION FAILS in members {member_ids}. "
                    "Upgrade those sections to next heavier. Add mid-span lateral nodes where L > 6 m."
                )
            if slender_fail:
                parts.append(
                    "  SLENDERNESS FAILS (KL/r > 200): Insert intermediate bracing nodes to halve those members."
                )
            feedback = "\n".join(parts)

        print("Optimization loop complete.")
        return best_graph, best_results

    # ------------------------------------------------------------------
    def build_three_variants(
        self,
        base_graph,
        base_results: dict,
        material_params: dict,
    ) -> list:
        """
        Produce 3 variants from the converged base design via section scaling.
        Runs FEA on MIN_COST and MIN_DISP; BALANCED reuses base_results.

        Returns
        -------
        list of dict: [{graph, fea, goal}, ...]  in order [MIN_COST, BALANCED, MIN_DISP]
        """
        # MIN_COST — lighter sections (2 steps down)
        g_cost = _scale_sections(base_graph, steps=-2)
        r_cost = _run_fea(g_cost, material_params)

        # MIN_DISP — heavier sections (2 steps up)
        g_perf = _scale_sections(base_graph, steps=+2)
        r_perf = _run_fea(g_perf, material_params)

        return [
            {"graph": g_cost,    "fea": r_cost,    "goal": "MIN_COST"},
            {"graph": base_graph, "fea": base_results, "goal": "BALANCED"},
            {"graph": g_perf,    "fea": r_perf,    "goal": "MIN_DISP"},
        ]


if __name__ == "__main__":
    print("Evolutionary Optimizer Ready.")
