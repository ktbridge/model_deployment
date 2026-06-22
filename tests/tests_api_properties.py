import pytest
import schemathesis
from fastapi.testclient import TestClient
from devops_ml.app import app

# 1. Link Schemathesis directly to your FastAPI app instance
schema = schemathesis.from_asgi("/openapi.json", app)

# 2. Automatically generate and run property-based tests
@schema.parametrize()
def test_api_endpoints(case):
    # This automatically sends hundreds of wild inputs to /predict and /
    response = case.call_asgi()
    
    # Assert that the server handles bad inputs gracefully (returning 422) 
    # instead of crashing completely with a 500 Internal Server Error
    case.validate_response(response)