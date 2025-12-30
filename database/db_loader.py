"""
Database loader for Streamlit dashboard
Loads predictions and recommendations from PostgreSQL
"""
import os
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime, timedelta

try:
    from database.db_connection import get_db_session, test_connection
    from database.db_operations import (
        get_latest_predictions,
        get_latest_optimization_run,
        get_optimization_recommendations_by_run,
        get_unimplemented_recommendations
    )
    from database.models import OptimizationRun, OptimizationRecommendation
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def is_database_available() -> bool:
    """Check if database is available and connected"""
    if not DB_AVAILABLE:
        return False
    try:
        # Try SQLAlchemy connection
        result = test_connection()
        if result:
            return True
    except Exception:
        pass
    
    # Fallback: Try direct psycopg2 connection
    try:
        from database.direct_connection import test_direct_connection
        success, _ = test_direct_connection()
        return success
    except Exception:
        return False


def load_recommendations_from_db(run_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load optimization recommendations from database"""
    if not is_database_available():
        return None
    
    try:
        with get_db_session() as session:
            if run_id is None:
                # Get latest run
                latest_run = get_latest_optimization_run(session)
                if not latest_run:
                    return None
                run_id = latest_run.run_id
            
            recommendations = get_optimization_recommendations_by_run(session, run_id)
            
            if not recommendations:
                return None
            
            # Convert to DataFrame
            data = []
            for rec in recommendations:
                data.append({
                    'window_id': rec.window_id,
                    'start_time': rec.start_time,
                    'end_time': rec.end_time,
                    'current_yard_util': rec.current_yard_util,
                    'recommended_yard_util': rec.recommended_yard_util,
                    'recommended_cranes': rec.recommended_cranes,
                    'current_delay_risk': rec.current_delay_risk,
                    'expected_risk_reduction': rec.expected_risk_reduction,
                    'current_truck_wait': rec.current_truck_wait,
                    'implemented': rec.implemented
                })
            
            return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading from database: {e}")
        return None


def load_optimization_summary_from_db() -> Optional[Dict]:
    """Load latest optimization run summary from database"""
    if not is_database_available():
        return None
    
    try:
        with get_db_session() as session:
            latest_run = get_latest_optimization_run(session)
            if not latest_run:
                return None
            
            return {
                'status': latest_run.status,
                'timestamp': latest_run.run_timestamp.strftime("%Y%m%d_%H%M%S"),
                'run_id': latest_run.run_id,
                'before': {
                    'avg_delay_risk': latest_run.before_avg_delay_risk,
                    'high_risk_windows': 0,  # Not stored separately
                    'avg_yard_util': 0.0,  # Not stored separately
                    'avg_truck_wait': 0.0,  # Not stored separately
                    'total_cranes_used': 0
                },
                'after': {
                    'avg_delay_risk': latest_run.after_avg_delay_risk,
                    'high_risk_windows': 0,
                    'avg_yard_util': 0.0,
                    'avg_truck_wait': 0.0,
                    'total_cranes_used': 0
                },
                'improvements': {
                    'delay_risk_reduction_pct': latest_run.delay_risk_reduction_pct or 0.0,
                    'high_risk_reduction': latest_run.high_risk_windows_reduced or 0,
                    'yard_util_improvement_pct': latest_run.yard_util_improvement_pct or 0.0,
                    'truck_wait_improvement_pct': 0.0
                },
                'config': {
                    'max_available_cranes': latest_run.max_available_cranes,
                    'min_cranes_per_window': latest_run.min_cranes_per_window,
                    'safe_yard_utilization': latest_run.safe_yard_utilization,
                    'max_yard_utilization': latest_run.max_yard_utilization
                }
            }
    except Exception as e:
        print(f"Error loading summary from database: {e}")
        return None


def get_available_optimization_runs() -> List[Dict]:
    """Get list of all optimization runs"""
    if not is_database_available():
        return []
    
    try:
        with get_db_session() as session:
            runs = session.query(OptimizationRun).order_by(
                OptimizationRun.run_timestamp.desc()
            ).limit(50).all()
            
            return [
                {
                    'run_id': run.run_id,
                    'timestamp': run.run_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'status': run.status,
                    'total_recommendations': run.total_recommendations
                }
                for run in runs
            ]
    except Exception:
        return []

