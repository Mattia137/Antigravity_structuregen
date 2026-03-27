import time
import random

class MockGraph:
    def __init__(self, num_nodes):
        self._nodes = [(i, {"coords": (random.random(), random.random(), random.random())}) for i in range(num_nodes)]

    def nodes(self, data=False):
        if data:
            return self._nodes
        return [n[0] for n in self._nodes]

def original_logic(graph):
    min_y = float('inf')
    for node_id, data in graph.nodes(data=True):
        coords = data["coords"]
        # simulate self.model.add_node(str(node_id), coords[0], coords[1], coords[2])
        if coords[1] < min_y:
            min_y = coords[1]

    y_vals = [data["coords"][1] for _, data in graph.nodes(data=True)]
    y_range = max(y_vals) - min_y if y_vals else 1.0
    return y_range

def optimized_logic(graph):
    min_y = float('inf')
    max_y = float('-inf')
    for node_id, data in graph.nodes(data=True):
        coords = data["coords"]
        # simulate self.model.add_node(str(node_id), coords[0], coords[1], coords[2])
        y = coords[1]
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    y_range = max_y - min_y if max_y != float('-inf') else 1.0
    return y_range

def benchmark():
    num_nodes = 1000000
    graph = MockGraph(num_nodes)

    print(f"Benchmarking with {num_nodes} nodes...")

    start = time.time()
    original_logic(graph)
    end = time.time()
    print(f"Original logic: {end - start:.4f} seconds")

    start = time.time()
    optimized_logic(graph)
    end = time.time()
    print(f"Optimized logic: {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark()
