"""
Test database connection and operations
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_connection import test_connection, get_database_info, init_database
from database.db_operations import (
    save_prediction, save_optimization_run, save_optimization_recommendations,
    get_latest_predictions, get_latest_optimization_run
)
from database.db_connection import get_db_session


def test_basic_operations():
    """Test basic database operations"""
    print("=" * 60)
    print("DATABASE OPERATIONS TEST")
    print("=" * 60)
    
    # Test connection
    print("\n[1/5] Testing connection...")
    if not test_connection():
        print("[ERROR] Connection failed!")
        return False
    print("[SUCCESS] Connection successful!")
    
    # Get database info
    print("\n[2/5] Database information:")
    info = get_database_info()
    print(f"   Host: {info['host']}")
    print(f"   Database: {info['database']}")
    print(f"   Tables: {', '.join(info.get('tables', []))}")
    
    # Test saving prediction
    print("\n[3/5] Testing save_prediction...")
    try:
        with get_db_session() as session:
            prediction = save_prediction(
                session=session,
                timestamp=datetime.utcnow(),
                time_window_start=datetime(2023, 1, 1, 0, 0),
                predicted_risk_class=1,
                probability_low=0.2,
                probability_medium=0.6,
                probability_high=0.2,
                model_type="TestModel",
                yard_utilization_ratio=0.75,
                avg_truck_wait_min=30.5
            )
            print(f"[SUCCESS] Saved prediction ID: {prediction.id}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    
    # Test querying predictions
    print("\n[4/5] Testing get_latest_predictions...")
    try:
        with get_db_session() as session:
            predictions = get_latest_predictions(session, limit=5)
            print(f"[SUCCESS] Retrieved {len(predictions)} predictions")
            if predictions:
                print(f"   Latest: {predictions[0].timestamp} - {predictions[0].predicted_risk_label}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    
    # Test optimization run
    print("\n[5/5] Testing save_optimization_run...")
    try:
        with get_db_session() as session:
            run = save_optimization_run(
                session=session,
                run_id="test_run_001",
                status="optimal",
                solver_used="Test",
                config={
                    'max_available_cranes': 20,
                    'min_cranes_per_window': 2,
                    'safe_yard_utilization': 0.80,
                    'max_yard_utilization': 0.95,
                    'risk_weight': 10.0,
                    'congestion_weight': 5.0,
                    'resource_cost_weight': 1.0
                },
                total_windows=10,
                total_recommendations=10,
                objective_value=100.5,
                impact_metrics={
                    'before': {'avg_delay_risk': 0.5},
                    'after': {'avg_delay_risk': 0.4},
                    'improvements': {
                        'delay_risk_reduction_pct': 20.0,
                        'high_risk_reduction': 2,
                        'yard_util_improvement_pct': 10.0
                    }
                }
            )
            print(f"[SUCCESS] Saved optimization run ID: {run.run_id}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_basic_operations()
    sys.exit(0 if success else 1)

