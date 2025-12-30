"""
Optimization service for resource optimization
Reuses existing optimization logic from Step 7
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import optimization functions from Step 7
from scripts.optimize_resources import (
    prepare_optimization_data,
    optimize_with_ortools,
    optimize_with_pulp,
    heuristic_optimization,
    calculate_impact_analysis
)

# Check solver availability
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


def optimize_resources(
    request_data: Dict,
    save_to_db: bool = False
) -> Dict:
    """
    Optimize resource allocation based on request data
    
    Args:
        request_data: Dictionary with optimization parameters
        save_to_db: Whether to save results to database
        
    Returns:
        Dictionary with optimization results
    """
    # Load base dataset for context
    data_file = project_root / "output" / "processed" / "ml_features_targets_regression_refined.csv"
    if not data_file.exists():
        raise FileNotFoundError("ML dataset not found. Run feature engineering first.")
    
    df = pd.read_csv(data_file, nrows=1000)  # Load sample for context
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Create synthetic data based on request
    time_window_hours = request_data.get('time_window_hours', 24)
    current_yard_util = request_data.get('current_yard_utilization', 0.85)
    current_delay_risk = request_data.get('current_delay_risk', 1.5)
    avg_truck_wait = request_data.get('avg_truck_wait_min', 45.0)
    
    # Create time windows
    now = datetime.utcnow()
    windows = []
    for i in range(max(1, time_window_hours // 6)):  # Create windows every 6 hours
        window_start = now + timedelta(hours=i * 6)
        window_end = window_start + timedelta(hours=6)
        
        windows.append({
            'window_id': i,
            'start_time': window_start,
            'end_time': window_end,
            'avg_yard_util': current_yard_util,
            'max_yard_util': min(1.0, current_yard_util + 0.1),
            'avg_truck_wait': avg_truck_wait,
            'max_truck_wait': avg_truck_wait + 10.0,
            'avg_delay_risk': current_delay_risk,
            'max_delay_risk': min(2, int(current_delay_risk) + 1),
            'high_risk_count': 1 if current_delay_risk >= 1.5 else 0,
            'n_records': 10
        })
    
    # Prepare optimization data
    opt_data = {
        'windows': windows,
        'time_col': 'timestamp',
        'total_windows': len(windows)
    }
    
    # Configuration
    config = {
        'max_available_cranes': request_data.get('max_available_cranes') or request_data.get('available_cranes', 20),
        'min_cranes_per_window': request_data.get('min_cranes_per_window', 2),
        'safe_yard_utilization': request_data.get('safe_yard_utilization', 0.80),
        'max_yard_utilization': request_data.get('max_yard_utilization', 0.95),
        'risk_weight': 10.0,
        'congestion_weight': 5.0,
        'resource_cost_weight': 1.0,
        'time_window_hours': time_window_hours
    }
    
    # Run optimization
    if ORTOOLS_AVAILABLE:
        solver_used = "OR-Tools"
        results = optimize_with_ortools(opt_data, config)
    elif PULP_AVAILABLE:
        solver_used = "PuLP"
        results = optimize_with_pulp(opt_data, config)
    else:
        solver_used = "Heuristic"
        results = heuristic_optimization(opt_data, config)
    
    # Calculate impact
    impact = calculate_impact_analysis(opt_data, results['recommendations'])
    
    # Format recommendations
    formatted_recommendations = []
    for rec in results['recommendations'][:1]:  # Return first recommendation as summary
        formatted_recommendations.append({
            'window_id': rec.get('window_id', 0),
            'start_time': rec.get('start_time', datetime.utcnow()),
            'end_time': rec.get('end_time', datetime.utcnow() + timedelta(hours=6)),
            'current_yard_util': rec.get('current_yard_util', current_yard_util),
            'recommended_yard_util': rec.get('recommended_yard_util', 0.75),
            'recommended_cranes': rec.get('recommended_cranes', config['max_available_cranes'] // 2),
            'current_delay_risk': rec.get('current_delay_risk', current_delay_risk),
            'expected_risk_reduction': rec.get('expected_risk_reduction', 0.0)
        })
    
    # Get summary metrics
    before_risk = impact['before'].get('avg_delay_risk', current_delay_risk)
    after_risk = impact['after'].get('avg_delay_risk', current_delay_risk * 0.6)
    improvement = impact['improvements'].get('delay_risk_reduction_pct', 0.0)
    
    # Get recommended values from first recommendation
    first_rec = formatted_recommendations[0] if formatted_recommendations else {}
    
    result = {
        'status': results.get('status', 'optimal'),
        'solver_used': solver_used,
        'recommended_cranes': first_rec.get('recommended_cranes', config['max_available_cranes'] // 2),
        'yard_utilization_target': first_rec.get('recommended_yard_util', 0.75),
        'delay_risk_before': float(before_risk),
        'delay_risk_after': float(after_risk),
        'improvement_percent': float(improvement),
        'recommendations': formatted_recommendations
    }
    
    # Save to database if requested
    if save_to_db:
        try:
            from api.database import get_db_session_for_api, save_optimization_run, save_optimization_recommendations
            from datetime import datetime
            
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            with get_db_session_for_api() as session:
                # Save optimization run
                opt_run = save_optimization_run(
                    session=session,
                    run_id=run_id,
                    status=result['status'],
                    solver_used=solver_used,
                    config=config,
                    total_windows=len(windows),
                    total_recommendations=len(formatted_recommendations),
                    objective_value=results.get('objective_value'),
                    impact_metrics=impact
                )
                
                # Save recommendations
                save_optimization_recommendations(
                    session=session,
                    recommendations=formatted_recommendations,
                    optimization_run_id=run_id,
                    optimization_status=result['status'],
                    objective_value=results.get('objective_value')
                )
                
                result['optimization_run_id'] = run_id
        except Exception as e:
            # Log error but don't fail the request
            print(f"Warning: Failed to save optimization to database: {e}")
    
    return result

