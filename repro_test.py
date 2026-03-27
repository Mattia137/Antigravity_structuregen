import os
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['networkx'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()

from src.ai_designer import AIDesigner

def test_missing_api_key():
    print("Testing missing API key...")
    # Clear relevant environment variables
    for var in ["GEMINI_AGENT_02", "GEMINI_AGENT_01", "GOOGLE_API_KEY"]:
        if var in os.environ:
            del os.environ[var]

    try:
        AIDesigner()
        print("FAIL: AIDesigner did not raise ValueError when API key is missing")
        return False
    except ValueError as e:
        print(f"SUCCESS: Caught expected ValueError: {e}")
        return True

def test_provided_api_key():
    print("\nTesting provided API key...")
    os.environ["GOOGLE_API_KEY"] = "test_key"

    try:
        designer = AIDesigner()
        if designer.api_key == "test_key":
            print("SUCCESS: API key correctly assigned")
            return True
        else:
            print(f"FAIL: API key was {designer.api_key}, expected 'test_key'")
            return False
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
        return False

if __name__ == "__main__":
    success = True
    success &= test_missing_api_key()
    success &= test_provided_api_key()

    if success:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
