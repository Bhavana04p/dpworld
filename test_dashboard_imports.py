"""
Test if dashboard can import database modules correctly
"""
import sys
from pathlib import Path

# Simulate what Streamlit does
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Testing imports...")
print(f"Project root: {project_root}")
print(f"Python path includes project root: {str(project_root) in sys.path}")

try:
    from database.db_connection import test_connection, get_database_info
    print("[SUCCESS] Imported database.db_connection")
    
    from database.db_loader import is_database_available
    print("[SUCCESS] Imported database.db_loader")
    
    # Test connection
    print("\nTesting connection...")
    result = test_connection()
    print(f"Connection test: {result}")
    
    if result:
        info = get_database_info()
        print(f"Database: {info.get('database')}")
        print(f"Tables: {len(info.get('tables', []))}")
        print("\n[SUCCESS] All imports and connection working!")
    else:
        print("\n[WARNING] Connection failed but imports work")
        
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"[ERROR] Other error: {e}")
    import traceback
    traceback.print_exc()

