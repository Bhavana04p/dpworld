"""
Quick script to view all new feature outputs
Run: python scripts/view_all_outputs.py
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 70)
print("VIEWING ALL NEW FEATURE OUTPUTS")
print("=" * 70)

# 1. Vessel Data Integration
print("\n[1] VESSEL DATA INTEGRATION")
try:
    vessel_file = project_root / "output" / "processed" / "ml_features_targets_regression_refined_with_vessels.csv"
    if vessel_file.exists():
        df = pd.read_csv(vessel_file, nrows=5)
        vessel_cols = [c for c in df.columns if c.startswith('vessel_')]
        ais_cols = [c for c in df.columns if c.startswith('ais_')]
        print(f"   [OK] Enhanced dataset found: {len(df.columns)} columns")
        print(f"   [OK] Vessel features: {len(vessel_cols)}")
        print(f"   [OK] AIS features: {len(ais_cols)}")
        print(f"   File: {vessel_file}")
    else:
        print(f"   [WARN] Enhanced dataset not found. Run: python scripts/integrate_vessel_data.py")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 2. Cost & Emission Analysis
print("\n[2] COST & EMISSION ANALYSIS")
try:
    from scripts.cost_emission_analysis import calculate_delay_cost
    cost = calculate_delay_cost("VSL100000", delay_hours=2.5)
    print(f"   [OK] Module loaded successfully")
    print(f"   Example cost (2.5h delay): ${cost['total_delay_cost_usd']:,.2f}")
    print(f"   CO2 emission: {cost['co2_emission_tonnes']:.2f} tonnes")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 3. Multi-Horizon Predictions
print("\n[3] MULTI-HORIZON PREDICTIONS")
try:
    models_dir = project_root / "output" / "models"
    models = ['random_forest_delay_risk_24h.joblib', 
              'random_forest_delay_risk_48h.joblib',
              'random_forest_delay_risk_72h.joblib']
    found_models = []
    for model in models:
        if (models_dir / model).exists():
            found_models.append(model.replace('random_forest_delay_risk_', '').replace('.joblib', ''))
    
    if found_models:
        print(f"   [OK] Models found: {', '.join(found_models)}")
        print(f"   Location: {models_dir}")
    else:
        print(f"   [WARN] Models not found. Run: python scripts/multi_horizon_prediction.py")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 4. Weather Data Integration
print("\n[4] WEATHER DATA INTEGRATION")
try:
    weather_file = project_root / "weather_data.csv"
    if weather_file.exists():
        df = pd.read_csv(weather_file, nrows=5)
        print(f"   [OK] Weather data file found")
        print(f"   File: {weather_file}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
    else:
        print(f"   [WARN] Weather data file not found")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 5. Heatmaps
print("\n[5] REAL-TIME HEATMAPS")
try:
    heatmap_dir = project_root / "output" / "heatmaps"
    if heatmap_dir.exists():
        heatmaps = list(heatmap_dir.glob('*.png'))
        if heatmaps:
            print(f"   [OK] Generated heatmaps: {len(heatmaps)}")
            print(f"   Location: {heatmap_dir}")
            print(f"   Files:")
            for hm in sorted(heatmaps)[-3:]:  # Show last 3
                print(f"      - {hm.name}")
        else:
            print(f"   [WARN] No heatmaps found. Run: python scripts/heatmap_generator.py")
    else:
        print(f"   [WARN] Heatmap directory not found. Run: python scripts/heatmap_generator.py")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 6. What-If Simulation
print("\n[6] WHAT-IF SIMULATION")
try:
    from scripts.what_if_simulator import simulate_berth_allocation
    current = {'yard_utilization_ratio': 0.85, 'available_cranes': 20, 'current_delay_risk': 1.5}
    proposed = {'cranes_per_berth': {1: 3, 2: 3}}
    result = simulate_berth_allocation(current, proposed)
    print(f"   [OK] Module loaded successfully")
    print(f"   Example simulation:")
    print(f"      Delay Risk: {result['current_state']['delay_risk']:.2f} -> {result['proposed_state']['delay_risk']:.2f}")
    print(f"      Net Benefit: ${result['cost_analysis']['net_benefit_usd']:,.2f}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 7. Optimization Results
print("\n[7] OPTIMIZATION RESULTS")
try:
    opt_dir = project_root / "output" / "optimization"
    if opt_dir.exists():
        recommendations = list(opt_dir.glob('recommendations_*.csv'))
        impacts = list(opt_dir.glob('impact_analysis_*.json'))
        if recommendations:
            print(f"   [OK] Optimization results found")
            print(f"   Location: {opt_dir}")
            print(f"   Recommendations: {len(recommendations)} files")
            print(f"   Impact analyses: {len(impacts)} files")
        else:
            print(f"   [WARN] No optimization results. Run: python scripts/optimize_resources.py")
    else:
        print(f"   [WARN] Optimization directory not found")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# 8. Database
print("\n[8] DATABASE RECORDS")
try:
    from database.db_loader import is_database_available
    if is_database_available():
        from database.db_operations import get_predictions_count, get_optimization_runs_count
        pred_count = get_predictions_count()
        opt_count = get_optimization_runs_count()
        print(f"   [OK] Database connected")
        print(f"   Predictions in DB: {pred_count}")
        print(f"   Optimization runs in DB: {opt_count}")
    else:
        print(f"   [WARN] Database not connected")
except Exception as e:
    print(f"   [WARN] Database check failed: {str(e)[:50]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nTo view detailed outputs:")
print("  1. Vessel Data: Load CSV file in pandas or Excel")
print("  2. Cost Analysis: Run calculate_delay_cost() function")
print("  3. Multi-Horizon: Use predict_multi_horizon() function")
print("  4. Weather: Load weather_data.csv")
print("  5. Heatmaps: Open PNG files in output/heatmaps/")
print("  6. What-If: Run simulate_berth_allocation() function")
print("  7. Optimization: View CSV/JSON files in output/optimization/")
print("\nFor detailed examples, see: HOW_TO_VIEW_OUTPUTS.md")
print("=" * 70)

