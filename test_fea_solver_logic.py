import math
import networkx as nx
import sys
from unittest.mock import MagicMock

# Mock Pynite since it's not installed
sys.modules['Pynite'] = MagicMock()
from Pynite import FEModel3D

# Mock config
sys.modules['config'] = MagicMock()
import config
config.DEFAULTS = {
    "Steel": {
        "Primary": "IPE_300",
        "Secondary": "HEA_200",
        "Diagonal": "Tubular_HSS_4x4x1/4"
    }
}

from src.fea_solver import FEASolver

def test_build_model_y_range():
    # Setup graph with nodes at different Y levels
    G = nx.Graph()
    G.add_node(1, coords=(0, 0, 0))
    G.add_node(2, coords=(0, 10, 0))
    G.add_node(3, coords=(0, 5, 0))

    material_params = {"type": "Steel"}
    solver = FEASolver(G, material_params)

    # We want to check if y_range is calculated correctly
    # Since y_range is a local variable in build_model, we might need to
    # check its effect, e.g., how tol is calculated and used in def_support.

    # Mock def_support to capture calls
    solver.model.def_support = MagicMock()

    solver.build_model()

    # min_y = 0, max_y = 10, y_range = 10
    # tol = max(0.5, 10 * 0.05) = max(0.5, 0.5) = 0.5
    # nodes with abs(y - 0) < 0.5 should be supported -> only node 1

    solver.model.def_support.assert_any_call('1', True, True, True, True, True, True)
    assert solver.model.def_support.call_count == 1
    print("Test passed: y_range correctly used for support detection.")

if __name__ == "__main__":
    test_build_model_y_range()
