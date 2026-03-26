import math
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['Pynite'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['config'] = MagicMock()

import networkx as nx
import numpy as np

# Define FEASolver mock context
class FEASolver:
    def __init__(self, structural_graph, material_params):
        self.graph = structural_graph
        self.model = MagicMock()
        self.material = material_params
        self._registered_sections = set()
        self._fallback_props = {}

    def _ensure_section(self, section_name, mat_type):
        pass

    def build_model(self):
        import config
        mat_type = self.material.get("type", "Steel")
        defaults = config.DEFAULTS.get(mat_type, {})
        role_default = {}

        # The actual logic from src/fea_solver.py that I modified
        min_y = float('inf')
        max_y = float('-inf')
        for node_id, data in self.graph.nodes(data=True):
            coords = data["coords"]
            self.model.add_node(str(node_id), coords[0], coords[1], coords[2])
            if coords[1] < min_y:
                min_y = coords[1]
            if coords[1] > max_y:
                max_y = coords[1]

        y_range = (max_y - min_y) if max_y != float('-inf') else 1.0
        tol = max(0.5, y_range * 0.05)

        supported_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if abs(data["coords"][1] - min_y) < tol:
                self.model.def_support(str(node_id), True, True, True, True, True, True)
                supported_nodes.append(node_id)
        return y_range, tol, supported_nodes

def test_build_model_logic():
    # Mock graph nodes
    nodes = [
        (1, {"coords": (0, 0, 0)}),
        (2, {"coords": (0, 10, 0)}),
        (3, {"coords": (0, 5, 0)}),
    ]
    graph = MagicMock()
    graph.nodes.return_value = nodes

    import config
    config.DEFAULTS = {"Steel": {}}

    solver = FEASolver(graph, {"type": "Steel"})
    y_range, tol, supported_nodes = solver.build_model()

    print(f"y_range: {y_range}, tol: {tol}, supported_nodes: {supported_nodes}")
    assert y_range == 10
    assert tol == 0.5
    assert supported_nodes == [1]
    print("Test passed: build_model logic is correct.")

if __name__ == "__main__":
    test_build_model_logic()
