import pytest
from unittest.mock import MagicMock, patch

def test_geometric_fallback_empty_data():
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {}
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result == expected

def test_geometric_fallback_zero_nodes():
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {"primary_nodes": []}
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result == expected

def test_geometric_fallback_one_node():
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {"primary_nodes": [{"id": 0, "x": 0, "y": 0, "z": 0}]}
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result == expected

def test_geometric_fallback_legacy_zero_vertices():
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {"vertices": []}
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result == expected

def test_geometric_fallback_legacy_one_vertex():
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {"vertices": [[0, 0, 0]]}
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result == expected

def test_geometric_fallback_two_nodes_no_fallback():
    # This ensures that with 2 nodes it DOES NOT trigger the generic cube fallback
    with patch.dict('sys.modules', {'networkx': MagicMock(), 'google': MagicMock(), 'google.genai': MagicMock(), 'numpy': MagicMock()}):
        from src.ai_designer import AIDesigner
        designer = AIDesigner()
        geometry_data = {
            "primary_nodes": [
                {"id": 0, "x": 0, "y": 0, "z": 0},
                {"id": 1, "x": 10, "y": 0, "z": 0}
            ],
            "primary_edges": [
                {"source": 0, "target": 1, "type": "primary_crease"}
            ]
        }
        # In a real environment, this might fail due to numpy being a mock,
        # but the logic threshold check happens BEFORE numpy is imported and used.
        result = designer._geometric_fallback(geometry_data)
        expected = designer._generic_cube_fallback()
        assert result != expected
        assert len(result["nodes"]) >= 2
        assert any(n["id"] == 0 for n in result["nodes"])
        assert any(n["id"] == 1 for n in result["nodes"])
