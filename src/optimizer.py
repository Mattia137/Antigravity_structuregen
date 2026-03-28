"""
Evolutionary Optimizer — 5-Phase Pipeline
==========================================

Phase 1  GEOMETRY    (mesh_processor.py)  — crease + floor nodes, story levels
Phase 2  TOPOLOGY    (ai_designer.py)     — Gemini outputs nodes + edges ONLY
Phase 3  SECTIONS    (section_sizer.py)   — deterministic R19/R20/R28 sizing
Phase 4  FEA + CHECK (fea_solver.py +     — PyNite + R39/R42 code checks
                      section_sizer.py)     revision loop ≤ max_iterations
Phase 5  VARIANTS    (section_sizer.py)   — ±2 ladder steps for 3 outputs

No section assignment happens in Gemini.  No extra API calls for variants.
"""
import networkx as nx
from src.section_sizer import (
    apply_sections,
    validate_structure,
    upgrade_failing_members,
    scale_sections,
)


# ── FEA helper ─────────────────────────────────────────────────────────────────

def _run_fea(graph: nx.Graph, material_params: dict) -> dict:
    from src.fea_solver import FEASolver
    solver = FEASolver(graph, material_params)
    solver.build_model()
    solver.apply_loads()
    return solver.solve_and_evaluate()


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer
# ══════════════════════════════════════════════════════════════════════════════

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
        Execute the full 5-phase pipeline and return the converged design.

        Returns
        -------
        (graph, fea_results) for the converged BALANCED design.
        Caller uses build_three_variants() to produce cost / balanced / disp variants.
        """
        mesh_desc = base_geometry.get("mesh_desc", {})
        feedback  = "Initial design. Follow structural rules exactly."
        best_graph   = None
        best_results = None

        for iteration in range(1, max_iterations + 1):
            print(f"\n[Iter {iteration}/{max_iterations}] Phases 2-4...")

            geom = dict(base_geometry)
            geom["optimization_feedback"] = feedback

            # ── Phase 2: Gemini topology ───────────────────────────────────────
            design_json = self.ai.request_base_design(geom)
            graph       = self.ai.construct_graph(design_json)
            best_graph  = graph

            # ── Phase 3: Deterministic section sizing ──────────────────────────
            apply_sections(graph, mesh_desc)

            # ── Phase 4a: FEA ─────────────────────────────────────────────────
            fea_results  = _run_fea(graph, material_params)
            best_results = fea_results

            # ── Phase 4b: Code checks (R39, R42) ──────────────────────────────
            check_report = validate_structure(graph, fea_results, mesh_desc)
            drift_DCR    = check_report.get("drift_DCR", 0.0)
            max_disp     = check_report.get("max_disp_m", 0.0)
            overall      = check_report.get("overall", "unknown")

            print(
                f"  FEA: δ_max={max_disp:.4f} m | drift_DCR={drift_DCR:.3f} | {overall}"
            )

            if overall == "pass":
                print(f"  ✓ All code checks PASS at iteration {iteration}.")
                break

            if iteration >= max_iterations:
                print(f"  ✗ Checks did not converge in {max_iterations} iterations.")
                break

            # ── Phase 4c: Section upgrades → re-FEA (no extra Gemini call) ────
            print("  Upgrading failing members (+1 ladder step)...")
            upgrade_failing_members(graph, check_report, steps=1)
            fea_results  = _run_fea(graph, material_params)
            best_results = fea_results

            # Build targeted feedback for the next Gemini call (if any)
            check_after = validate_structure(graph, fea_results, mesh_desc)
            if check_after["overall"] == "pass":
                print("  ✓ Checks passed after section upgrade.")
                break

            drift_fail    = check_after["drift_DCR"] > 1.0
            slender_fails = [
                c for c in check_after["checks"]
                if c["status"] == "fail" and "slenderness" in c["type"]
            ]
            parts = [
                f"ITERATION {iteration} REVISION:",
                f"  δ_max={check_after['max_disp_m']:.4f} m  drift_DCR={check_after['drift_DCR']:.3f}",
            ]
            if drift_fail:
                parts.append(
                    "  DRIFT FAILS: Add X-bracing in ALL perimeter bays. "
                    "Upgrade column sections. Add mid-height bracing on columns > 6 m."
                )
            if slender_fails:
                parts.append(
                    "  SLENDERNESS FAILS: Insert intermediate bracing nodes to halve long members."
                )
            feedback = "\n".join(parts)

        print("Optimization loop complete.")
        return best_graph, best_results

    # ------------------------------------------------------------------
    def build_three_variants(
        self,
        base_graph: nx.Graph,
        base_results: dict,
        material_params: dict,
    ) -> list:
        """
        Phase 5: Produce 3 variants by scaling sections ±2 steps on the ladder.
        Runs FEA on MIN_COST and MIN_DISP; BALANCED reuses base_results.

        Returns
        -------
        [{"graph": G, "fea": results, "goal": str}, ...]
        Order: MIN_COST, BALANCED, MIN_DISP
        """
        g_cost = scale_sections(base_graph, steps=-2)
        r_cost = _run_fea(g_cost, material_params)

        g_perf = scale_sections(base_graph, steps=+2)
        r_perf = _run_fea(g_perf, material_params)

        return [
            {"graph": g_cost,    "fea": r_cost,    "goal": "MIN_COST"},
            {"graph": base_graph, "fea": base_results, "goal": "BALANCED"},
            {"graph": g_perf,    "fea": r_perf,    "goal": "MIN_DISP"},
        ]


if __name__ == "__main__":
    print("Evolutionary Optimizer (5-phase pipeline) ready.")
