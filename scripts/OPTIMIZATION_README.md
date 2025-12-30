# Step 7: Prescriptive Optimization Module

## Overview

This module converts the predictive analytics system into a **prescriptive** system by providing actionable recommendations for port operations.

## Objective

**Minimize**: Total Delay Risk + Congestion Penalty + Resource Cost

**Decision Variables**:
- `crane_allocation[w]`: Number of cranes allocated to time window `w` (integer)
- `yard_target_util[w]`: Target yard utilization for window `w` (continuous, 0-1)

## Constraints

1. **Total Crane Constraint**: Sum of all crane allocations ≤ available cranes × number of windows
2. **Minimum Service Level**: High-risk windows (delay_risk ≥ 2) must have at least `min_cranes + 2`
3. **Yard Utilization Bounds**: Target utilization between 70% of safe threshold and maximum allowed
4. **Congestion Penalty**: Penalizes yard utilization exceeding safe threshold (80%)

## Installation

### Option 1: OR-Tools (Preferred)
```bash
pip install ortools
```

### Option 2: PuLP (Alternative)
```bash
pip install pulp
```

### Option 3: Heuristic (No Solver Required)
If neither solver is available, the script will use a rule-based heuristic approach.

## Usage

### Basic Usage
```bash
python scripts/optimize_resources.py
```

### Configuration

Edit the `config` dictionary in `main()` function:

```python
config = {
    'max_available_cranes': 20,          # Maximum cranes available
    'min_cranes_per_window': 2,         # Minimum cranes per window
    'safe_yard_utilization': 0.80,       # Safe yard utilization threshold
    'max_yard_utilization': 0.95,        # Maximum allowed utilization
    'risk_weight': 10.0,                 # Weight for delay risk penalty
    'congestion_weight': 5.0,            # Weight for congestion penalty
    'resource_cost_weight': 1.0,         # Weight for resource cost
    'time_window_hours': 24              # Time window size in hours
}
```

## Output Files

Results are saved to `output/optimization/`:

1. **`recommendations_YYYYMMDD_HHMMSS.csv`**
   - Detailed recommendations per time window
   - Columns: window_id, start_time, end_time, current_yard_util, recommended_yard_util, recommended_cranes, current_delay_risk, expected_risk_reduction

2. **`impact_analysis_YYYYMMDD_HHMMSS.json`**
   - Before/after impact metrics
   - Configuration used
   - Optimization status

3. **`summary_YYYYMMDD_HHMMSS.txt`**
   - Human-readable summary report
   - Before/after comparison
   - Improvement metrics

## Mathematical Formulation

### Objective Function
```
Minimize: Σ [resource_cost_weight × cranes[w] + 
             risk_weight × delay_risk[w] × (1 - yard_util[w]) + 
             congestion_weight × max(0, yard_util[w] - safe_threshold) × 10]
```

### Constraints
```
Σ cranes[w] ≤ max_cranes × n_windows
cranes[w] ≥ min_cranes + 2  (if delay_risk[w] ≥ 2)
safe_threshold × 0.7 ≤ yard_util[w] ≤ max_yard_util
```

## Integration with Dashboard

The Streamlit dashboard automatically loads the latest optimization results:
- Navigate to **"🎯 Optimization & Recommendations"** page
- View summary metrics, before/after comparison, and detailed recommendations
- Download results as CSV

## Example Output

```
BEFORE OPTIMIZATION:
  Average Delay Risk: 0.850
  High Risk Windows: 15
  Average Yard Utilization: 87.50%
  Average Truck Wait: 45.30 min

AFTER OPTIMIZATION:
  Average Delay Risk: 0.620
  High Risk Windows: 8
  Average Yard Utilization: 78.20%
  Average Truck Wait: 40.77 min
  Total Cranes Used: 180

IMPROVEMENTS:
  Delay Risk Reduction: 27.06%
  High Risk Windows Reduced: 7
  Yard Utilization Improvement: 10.63%
  Truck Wait Improvement: 10.00%
```

## Troubleshooting

### "No solver available"
- Install OR-Tools: `pip install ortools`
- Or install PuLP: `pip install pulp`
- The script will fall back to heuristic optimization if no solver is found

### "Infeasible solution"
- Check constraints are realistic
- Reduce `min_cranes_per_window` or increase `max_available_cranes`
- Verify data quality

### "No data found"
- Ensure `output/processed/ml_features_targets_regression_refined.parquet` exists
- Run data processing and feature engineering steps first

## Next Steps

1. **Tune Weights**: Adjust `risk_weight`, `congestion_weight`, and `resource_cost_weight` based on operational priorities
2. **Refine Constraints**: Add more realistic operational constraints (e.g., crane movement costs, berth availability)
3. **Real-time Integration**: Connect to live port data for real-time optimization
4. **Multi-objective Optimization**: Consider additional objectives (e.g., cost, emissions)

