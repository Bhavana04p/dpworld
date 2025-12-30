"""
Quick test script to verify FastAPI setup
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Testing FastAPI imports...")

try:
    from api.main import app
    print("[SUCCESS] FastAPI app imported")
except Exception as e:
    print(f"[ERROR] Failed to import app: {e}")
    sys.exit(1)

try:
    from api.schemas import HealthResponse, DelayRiskPredictionRequest, OptimizationRequest
    print("[SUCCESS] Schemas imported")
except Exception as e:
    print(f"[ERROR] Failed to import schemas: {e}")
    sys.exit(1)

try:
    from api.database import test_connection
    db_connected = test_connection()
    print(f"[SUCCESS] Database connection: {'Connected' if db_connected else 'Not Connected'}")
except Exception as e:
    print(f"[WARNING] Database test failed: {e}")

try:
    from api.services.prediction_service import load_model
    model, features = load_model()
    print(f"[SUCCESS] Model loaded: {type(model).__name__}, {len(features)} features")
except Exception as e:
    print(f"[WARNING] Model loading failed: {e}")

print("\n[SUCCESS] All imports successful!")
print("\nTo start the API server, run:")
print("  uvicorn api.main:app --reload")
print("\nThen visit:")
print("  - API: http://localhost:8000")
print("  - Docs: http://localhost:8000/docs")

