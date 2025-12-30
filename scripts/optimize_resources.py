"""
Step 7: Prescriptive Optimization Module
Converts predictive analytics to prescriptive recommendations

Objective: Minimize delay risk and congestion by optimizing:
- Crane allocation per time window
- Yard utilization thresholds
- Resource allocation decisions

Uses OR-Tools for optimization (falls back to PuLP if OR-Tools unavailable)
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try OR-Tools first, fallback to PuLP
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
        print("Warning: Neither OR-Tools nor PuLP available. Optimization will use heuristic approach.")

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_DIR / "output" / "processed"
OUTPUT_DIR = PROJECT_DIR / "output" / "optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PARQUET = PROC_DIR / "ml_features_targets_regression_refined.parquet"
DATA_CSV = PROC_DIR / "ml_features_targets_regression_refined.csv"
TARGET = "delay_risk_24h"


def load_data() -> pd.DataFrame:
    """Load the ML dataset"""
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_CSV.exists():
        df = pd.read_csv(DATA_CSV, low_memory=False)
    else:
        raise FileNotFoundError("ML dataset not found. Run feature engineering first.")
    
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def prepare_optimization_data(df: pd.DataFrame, time_window_hours: int = 24) -> Dict:
    """
    Prepare data for optimization
    
    Args:
        df: Input dataframe with predictions and features
        time_window_hours: Time window for optimization (default 24 hours)
    
    Returns:
        Dictionary with prepared data for optimization
    """
    # Ensure time column exists
    time_cols = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    if not time_cols:
        raise ValueError("No time column found in dataset")
    
    time_col = time_cols[0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Extract key features
    required_cols = ["yard_utilization_ratio", "avg_truck_wait_min", TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Group by time windows - use time-based grouping instead of pd.cut
    # Create time windows based on hours
    df = df.sort_values(time_col).reset_index(drop=True)
    df['time_window'] = ((df[time_col] - df[time_col].min()).dt.total_seconds() / 3600 / time_window_hours).astype(int)
    
    # Aggregate by time window
    window_data = []
    for window_id in df['time_window'].unique():
        window_df = df[df['time_window'] == window_id]
        if len(window_df) == 0:
            continue
        
        window_data.append({
            'window_id': int(window_id),
            'start_time': window_df[time_col].min(),
            'end_time': window_df[time_col].max(),
            'avg_yard_util': float(window_df['yard_utilization_ratio'].mean()),
            'max_yard_util': float(window_df['yard_utilization_ratio'].max()),
            'avg_truck_wait': float(window_df['avg_truck_wait_min'].mean()),
            'max_truck_wait': float(window_df['avg_truck_wait_min'].max()),
            'avg_delay_risk': float(window_df[TARGET].mean()),
            'max_delay_risk': int(window_df[TARGET].max()),
            'high_risk_count': int((window_df[TARGET] == 2).sum()),
            'n_records': len(window_df)
        })
    
    return {
        'windows': window_data,
        'time_col': time_col,
        'total_windows': len(window_data)
    }


def optimize_with_ortools(data: Dict, config: Dict) -> Dict:
    """
    Optimize using OR-Tools
    
    Decision Variables:
    - crane_allocation[w]: Number of cranes allocated to window w (integer)
    - yard_target_util[w]: Target yard utilization for window w (continuous, 0-1)
    
    Objective:
    Minimize: Total delay risk + congestion penalty + resource cost
    
    Constraints:
    - Total cranes <= available cranes
    - Yard utilization <= yard capacity
    - Minimum service level maintained
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Failed to create OR-Tools solver")
    
    windows = data['windows']
    n_windows = len(windows)
    
    # Parameters from config
    max_cranes = config.get('max_available_cranes', 20)
    min_cranes_per_window = config.get('min_cranes_per_window', 2)
    safe_yard_util = config.get('safe_yard_utilization', 0.80)
    max_yard_util = config.get('max_yard_utilization', 0.95)
    risk_weight = config.get('risk_weight', 10.0)
    congestion_weight = config.get('congestion_weight', 5.0)
    resource_cost_weight = config.get('resource_cost_weight', 1.0)
    
    # Decision variables
    crane_allocation = {}
    yard_target_util = {}
    congestion_excess = {}
    
    for i, w in enumerate(windows):
        # Crane allocation: integer, between min and max
        crane_allocation[i] = solver.IntVar(
            min_cranes_per_window, max_cranes,
            f'cranes_{i}'
        )
        # Yard target utilization: continuous, between safe and max
        yard_target_util[i] = solver.NumVar(
            safe_yard_util * 0.7, max_yard_util,
            f'yard_util_{i}'
        )
        # Congestion excess: auxiliary variable for max(0, yard_util - safe_util)
        congestion_excess[i] = solver.NumVar(
            0.0, max_yard_util,
            f'congestion_excess_{i}'
        )
    
    # Constraints
    # 1. Total crane constraint
    solver.Add(
        sum(crane_allocation[i] for i in range(n_windows)) <= max_cranes * n_windows
    )
    
    # 2. Per-window minimum service level
    for i, w in enumerate(windows):
        # Ensure minimum cranes for high-risk windows
        if w['max_delay_risk'] >= 2:
            solver.Add(crane_allocation[i] >= min_cranes_per_window + 2)
        # Congestion excess constraint: congestion_excess >= yard_util - safe_util
        solver.Add(congestion_excess[i] >= yard_target_util[i] - safe_yard_util)
    
    # Objective function
    # Minimize: risk penalty + congestion penalty + resource cost
    objective = solver.Objective()
    
    for i, w in enumerate(windows):
        # Risk penalty: higher for high delay risk
        risk_penalty_coeff = risk_weight * w['avg_delay_risk']
        
        # Congestion penalty: based on yard utilization exceeding safe threshold
        congestion_penalty_coeff = congestion_weight * 10
        
        # Resource cost: cost of allocating cranes
        resource_cost_coeff = resource_cost_weight
        
        # Set coefficients
        objective.SetCoefficient(crane_allocation[i], resource_cost_coeff)
        # Risk penalty: risk_weight * delay_risk * (1 - yard_util) = risk_weight * delay_risk - risk_weight * delay_risk * yard_util
        objective.SetCoefficient(yard_target_util[i], -risk_penalty_coeff)
        objective.SetCoefficient(congestion_excess[i], congestion_penalty_coeff)
        # Add constant term for risk_weight * delay_risk (doesn't affect optimization)
    
    objective.SetMinimization()
    
    # Solve
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        results = {
            'status': 'optimal',
            'objective_value': solver.Objective().Value(),
            'recommendations': []
        }
        
        for i, w in enumerate(windows):
            results['recommendations'].append({
                'window_id': w['window_id'],
                'start_time': str(w['start_time']),
                'end_time': str(w['end_time']),
                'current_yard_util': w['avg_yard_util'],
                'recommended_yard_util': yard_target_util[i].solution_value(),
                'recommended_cranes': int(crane_allocation[i].solution_value()),
                'current_delay_risk': w['avg_delay_risk'],
                'expected_risk_reduction': w['avg_delay_risk'] * (1.0 - yard_target_util[i].solution_value() / max(0.01, w['avg_yard_util']))
            })
        
        return results
    else:
        return {'status': 'infeasible', 'recommendations': []}


def optimize_with_pulp(data: Dict, config: Dict) -> Dict:
    """Optimize using PuLP (fallback)"""
    problem = pulp.LpProblem("Port_Resource_Optimization", pulp.LpMinimize)
    
    windows = data['windows']
    n_windows = len(windows)
    
    # Parameters
    max_cranes = config.get('max_available_cranes', 20)
    min_cranes_per_window = config.get('min_cranes_per_window', 2)
    safe_yard_util = config.get('safe_yard_utilization', 0.80)
    max_yard_util = config.get('max_yard_utilization', 0.95)
    risk_weight = config.get('risk_weight', 10.0)
    congestion_weight = config.get('congestion_weight', 5.0)
    resource_cost_weight = config.get('resource_cost_weight', 1.0)
    
    # Decision variables
    crane_allocation = {}
    yard_target_util = {}
    congestion_excess = {}
    
    for i, w in enumerate(windows):
        crane_allocation[i] = pulp.LpVariable(
            f'cranes_{i}', lowBound=min_cranes_per_window,
            upBound=max_cranes, cat='Integer'
        )
        yard_target_util[i] = pulp.LpVariable(
            f'yard_util_{i}', lowBound=safe_yard_util * 0.7,
            upBound=max_yard_util, cat='Continuous'
        )
        congestion_excess[i] = pulp.LpVariable(
            f'congestion_excess_{i}', lowBound=0, cat='Continuous'
        )
    
    # Constraints
    problem += sum(crane_allocation[i] for i in range(n_windows)) <= max_cranes * n_windows
    
    for i, w in enumerate(windows):
        if w['max_delay_risk'] >= 2:
            problem += crane_allocation[i] >= min_cranes_per_window + 2
        # Congestion excess constraint
        problem += congestion_excess[i] >= yard_target_util[i] - safe_yard_util
    
    # Objective
    objective = 0
    for i, w in enumerate(windows):
        risk_penalty = risk_weight * w['avg_delay_risk'] * (1.0 - yard_target_util[i])
        congestion_penalty = congestion_weight * congestion_excess[i] * 10
        resource_cost = resource_cost_weight * crane_allocation[i]
        objective += resource_cost + risk_penalty + congestion_penalty
    
    problem += objective
    
    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if problem.status == pulp.LpStatusOptimal:
        results = {
            'status': 'optimal',
            'objective_value': pulp.value(problem.objective),
            'recommendations': []
        }
        
        for i, w in enumerate(windows):
            results['recommendations'].append({
                'window_id': w['window_id'],
                'start_time': str(w['start_time']),
                'end_time': str(w['end_time']),
                'current_yard_util': w['avg_yard_util'],
                'recommended_yard_util': yard_target_util[i].varValue,
                'recommended_cranes': int(crane_allocation[i].varValue),
                'current_delay_risk': w['avg_delay_risk'],
                'expected_risk_reduction': w['avg_delay_risk'] * (1.0 - yard_target_util[i].varValue / max(0.01, w['avg_yard_util']))
            })
        
        return results
    else:
        return {'status': 'infeasible', 'recommendations': []}


def heuristic_optimization(data: Dict, config: Dict) -> Dict:
    """
    Heuristic optimization when no solver available
    Simple rule-based approach
    """
    windows = data['windows']
    max_cranes = config.get('max_available_cranes', 20)
    min_cranes = config.get('min_cranes_per_window', 2)
    safe_yard_util = config.get('safe_yard_utilization', 0.80)
    
    recommendations = []
    total_cranes_used = 0
    
    # Sort windows by risk (highest first)
    sorted_windows = sorted(windows, key=lambda x: x['avg_delay_risk'], reverse=True)
    
    for w in sorted_windows:
        # Allocate cranes based on risk
        if w['max_delay_risk'] >= 2:  # High risk
            cranes = min(max_cranes, min_cranes + 4)
        elif w['max_delay_risk'] == 1:  # Medium risk
            cranes = min(max_cranes, min_cranes + 2)
        else:  # Low risk
            cranes = min_cranes
        
        # Adjust for remaining capacity
        if total_cranes_used + cranes > max_cranes * len(windows):
            cranes = max(min_cranes, max_cranes * len(windows) - total_cranes_used)
        
        total_cranes_used += cranes
        
        # Target yard utilization: reduce if current is high
        current_util = w['avg_yard_util']
        if current_util > safe_yard_util:
            target_util = safe_yard_util * 0.9
        else:
            target_util = min(safe_yard_util, current_util * 1.1)
        
        recommendations.append({
            'window_id': w['window_id'],
            'start_time': str(w['start_time']),
            'end_time': str(w['end_time']),
            'current_yard_util': current_util,
            'recommended_yard_util': target_util,
            'recommended_cranes': cranes,
            'current_delay_risk': w['avg_delay_risk'],
            'expected_risk_reduction': w['avg_delay_risk'] * (1.0 - target_util / max(0.01, current_util))
        })
    
    return {
        'status': 'heuristic',
        'objective_value': sum(w['avg_delay_risk'] for w in windows),
        'recommendations': recommendations
    }


def calculate_impact_analysis(data: Dict, recommendations: List[Dict]) -> Dict:
    """Calculate before/after impact metrics"""
    windows = data['windows']
    
    # Before metrics
    before_metrics = {
        'avg_delay_risk': np.mean([w['avg_delay_risk'] for w in windows]),
        'high_risk_windows': sum(1 for w in windows if w['max_delay_risk'] >= 2),
        'avg_yard_util': np.mean([w['avg_yard_util'] for w in windows]),
        'avg_truck_wait': np.mean([w['avg_truck_wait'] for w in windows]),
        'total_cranes_used': 0  # Will be calculated from recommendations
    }
    
    # After metrics (estimated)
    after_metrics = {
        'avg_delay_risk': np.mean([r['current_delay_risk'] - r['expected_risk_reduction'] for r in recommendations]),
        'high_risk_windows': sum(1 for r in recommendations if (r['current_delay_risk'] - r['expected_risk_reduction']) >= 2),
        'avg_yard_util': np.mean([r['recommended_yard_util'] for r in recommendations]),
        'avg_truck_wait': before_metrics['avg_truck_wait'] * 0.9,  # Estimated improvement
        'total_cranes_used': sum(r['recommended_cranes'] for r in recommendations)
    }
    
    # Improvements
    improvements = {
        'delay_risk_reduction_pct': ((before_metrics['avg_delay_risk'] - after_metrics['avg_delay_risk']) / 
                                     max(0.01, before_metrics['avg_delay_risk'])) * 100,
        'high_risk_reduction': before_metrics['high_risk_windows'] - after_metrics['high_risk_windows'],
        'yard_util_improvement_pct': ((before_metrics['avg_yard_util'] - after_metrics['avg_yard_util']) / 
                                     max(0.01, before_metrics['avg_yard_util'])) * 100,
        'truck_wait_improvement_pct': ((before_metrics['avg_truck_wait'] - after_metrics['avg_truck_wait']) / 
                                       max(0.01, before_metrics['avg_truck_wait'])) * 100
    }
    
    return {
        'before': before_metrics,
        'after': after_metrics,
        'improvements': improvements
    }


def save_results(results: Dict, impact: Dict, config: Dict, output_dir: Path):
    """Save optimization results to files and database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    
    # Save recommendations as CSV
    recommendations_df = pd.DataFrame(results['recommendations'])
    recommendations_path = output_dir / f"recommendations_{timestamp}.csv"
    recommendations_df.to_csv(recommendations_path, index=False)
    
    # Save impact analysis as JSON
    impact_path = output_dir / f"impact_analysis_{timestamp}.json"
    with open(impact_path, 'w') as f:
        json.dump({
            'impact': impact,
            'config': config,
            'optimization_status': results['status'],
            'timestamp': timestamp
        }, f, indent=2, default=str)
    
    # Save summary report
    summary_path = output_dir / f"summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PORT RESOURCE OPTIMIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Optimization Status: {results['status']}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("BEFORE OPTIMIZATION:\n")
        f.write(f"  Average Delay Risk: {impact['before']['avg_delay_risk']:.3f}\n")
        f.write(f"  High Risk Windows: {impact['before']['high_risk_windows']}\n")
        f.write(f"  Average Yard Utilization: {impact['before']['avg_yard_util']:.2%}\n")
        f.write(f"  Average Truck Wait: {impact['before']['avg_truck_wait']:.2f} min\n\n")
        f.write("AFTER OPTIMIZATION:\n")
        f.write(f"  Average Delay Risk: {impact['after']['avg_delay_risk']:.3f}\n")
        f.write(f"  High Risk Windows: {impact['after']['high_risk_windows']}\n")
        f.write(f"  Average Yard Utilization: {impact['after']['avg_yard_util']:.2%}\n")
        f.write(f"  Average Truck Wait: {impact['after']['avg_truck_wait']:.2f} min\n")
        f.write(f"  Total Cranes Used: {impact['after']['total_cranes_used']}\n\n")
        f.write("IMPROVEMENTS:\n")
        f.write(f"  Delay Risk Reduction: {impact['improvements']['delay_risk_reduction_pct']:.2f}%\n")
        f.write(f"  High Risk Windows Reduced: {impact['improvements']['high_risk_reduction']}\n")
        f.write(f"  Yard Utilization Improvement: {impact['improvements']['yard_util_improvement_pct']:.2f}%\n")
        f.write(f"  Truck Wait Improvement: {impact['improvements']['truck_wait_improvement_pct']:.2f}%\n")
    
    print(f"\n[SUCCESS] Optimization results saved:")
    print(f"   - Recommendations: {recommendations_path}")
    print(f"   - Impact Analysis: {impact_path}")
    print(f"   - Summary: {summary_path}")
    
    # Save to database
    try:
        from database.db_connection import get_db_session
        from database.db_operations import save_optimization_run, save_optimization_recommendations
        
        solver_used = "OR-Tools" if ORTOOLS_AVAILABLE else ("PuLP" if PULP_AVAILABLE else "Heuristic")
        
        with get_db_session() as session:
            # Save optimization run
            opt_run = save_optimization_run(
                session=session,
                run_id=run_id,
                status=results['status'],
                solver_used=solver_used,
                config=config,
                total_windows=len(results['recommendations']),
                total_recommendations=len(results['recommendations']),
                objective_value=results.get('objective_value'),
                impact_metrics=impact
            )
            
            # Save recommendations
            save_optimization_recommendations(
                session=session,
                recommendations=results['recommendations'],
                optimization_run_id=run_id,
                optimization_status=results['status'],
                objective_value=results.get('objective_value')
            )
            
            print(f"   - Database: Saved to PostgreSQL (run_id: {run_id})")
    except ImportError:
        print("   - Database: Skipped (database module not available)")
    except Exception as e:
        print(f"   - Database: Failed to save ({str(e)})")


def main():
    """Main optimization function"""
    print("=" * 60)
    print("PORT RESOURCE OPTIMIZATION MODULE")
    print("=" * 60)
    
    # Load data
    print("\n[INFO] Loading data...")
    df = load_data()
    print(f"   Loaded {len(df):,} records")
    
    # Configuration
    config = {
        'max_available_cranes': 20,
        'min_cranes_per_window': 2,
        'safe_yard_utilization': 0.80,
        'max_yard_utilization': 0.95,
        'risk_weight': 10.0,
        'congestion_weight': 5.0,
        'resource_cost_weight': 1.0,
        'time_window_hours': 24
    }
    
    print("\n[CONFIG] Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Prepare data
    print("\n[INFO] Preparing optimization data...")
    data = prepare_optimization_data(df, config['time_window_hours'])
    print(f"   Created {data['total_windows']} time windows")
    
    # Run optimization
    print("\n[OPTIMIZE] Running optimization...")
    if ORTOOLS_AVAILABLE:
        print("   Using OR-Tools solver")
        results = optimize_with_ortools(data, config)
    elif PULP_AVAILABLE:
        print("   Using PuLP solver")
        results = optimize_with_pulp(data, config)
    else:
        print("   Using heuristic approach (no solver available)")
        results = heuristic_optimization(data, config)
    
    print(f"   Status: {results['status']}")
    print(f"   Generated {len(results['recommendations'])} recommendations")
    
    # Calculate impact
    print("\n[ANALYSIS] Calculating impact analysis...")
    impact = calculate_impact_analysis(data, results['recommendations'])
    
    print("\n[SUMMARY] IMPACT SUMMARY:")
    print(f"   Delay Risk Reduction: {impact['improvements']['delay_risk_reduction_pct']:.2f}%")
    print(f"   High Risk Windows Reduced: {impact['improvements']['high_risk_reduction']}")
    print(f"   Yard Utilization Improvement: {impact['improvements']['yard_util_improvement_pct']:.2f}%")
    
    # Save results
    print("\n[SAVE] Saving results...")
    save_results(results, impact, config, OUTPUT_DIR)
    
    print("\n[SUCCESS] Optimization complete!")
    return results, impact


if __name__ == "__main__":
    main()

