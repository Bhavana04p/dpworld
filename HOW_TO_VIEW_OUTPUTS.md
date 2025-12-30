# 📊 How to View Outputs of New Features

## Quick Start Guide

---

## 1. 🚢 **Vessel Data Integration**

### View Enhanced Dataset
```bash
# The enhanced dataset is saved here:
output/processed/ml_features_targets_regression_refined_with_vessels.csv
output/processed/ml_features_targets_regression_refined_with_vessels.parquet
```

### Check Vessel Features
```python
import pandas as pd

# Load enhanced dataset
df = pd.read_csv('output/processed/ml_features_targets_regression_refined_with_vessels.csv')

# View vessel features (columns starting with 'vessel_')
vessel_cols = [c for c in df.columns if c.startswith('vessel_')]
print(f"Vessel features: {len(vessel_cols)}")
print(vessel_cols)

# View AIS features (columns starting with 'ais_')
ais_cols = [c for c in df.columns if c.startswith('ais_')]
print(f"\nAIS features: {len(ais_cols)}")
print(ais_cols)

# View summary statistics
print("\nVessel Features Summary:")
print(df[vessel_cols].describe())
```

### Run Integration Script
```bash
python scripts/integrate_vessel_data.py
```
**Output**: Console shows number of features added and file locations

---

## 2. 💰 **Cost & Emission Analysis**

### Calculate Costs for a Prediction
```python
from scripts.cost_emission_analysis import calculate_delay_cost

# Calculate cost for a vessel with delay
cost_info = calculate_delay_cost(
    vessel_id="VSL100000",
    delay_hours=2.5
)

print("Cost Breakdown:")
print(f"  Fuel Cost: ${cost_info['fuel_cost_usd']:,.2f}")
print(f"  Penalty Cost: ${cost_info['penalty_cost_usd']:,.2f}")
print(f"  Opportunity Cost: ${cost_info['opportunity_cost_usd']:,.2f}")
print(f"  Total Cost: ${cost_info['total_delay_cost_usd']:,.2f}")
print(f"\nEmissions:")
print(f"  CO2 Emission: {cost_info['co2_emission_tonnes']:.2f} tonnes")
```

### Analyze Predictions with Costs
```python
from scripts.cost_emission_analysis import calculate_prediction_cost_impact
import pandas as pd

# Load predictions
predictions = pd.DataFrame({
    'predicted_risk_class': [0, 1, 2],
    'vessel_id': ['VSL100000', 'VSL100001', 'VSL100002']
})

# Calculate costs
cost_analysis = calculate_prediction_cost_impact(predictions)
print(cost_analysis[['estimated_delay_hours', 'total_delay_cost_usd', 'co2_emission_tonnes']])
```

### Run Cost Analysis Script
```bash
python scripts/cost_emission_analysis.py
```
**Output**: Test calculation showing cost breakdown

---

## 3. ⏰ **Multi-Horizon Predictions (24h, 48h, 72h)**

### View Trained Models
```bash
# Models are saved here:
output/models/random_forest_delay_risk_24h.joblib
output/models/random_forest_delay_risk_48h.joblib
output/models/random_forest_delay_risk_72h.joblib
```

### Make Multi-Horizon Predictions
```python
from api.services.multi_horizon_service import predict_multi_horizon

# Prepare features (example)
features = {
    'yard_utilization_ratio': 0.85,
    'crane_utilization_ratio': 0.75,
    'avg_truck_wait_min': 45.0,
    'temperature_c': 25.0,
    'wind_speed_mps': 15.0,
    # ... add other features
}

# Predict for all horizons
predictions = predict_multi_horizon(features)

print("Multi-Horizon Predictions:")
for horizon in ['24h', '48h', '72h']:
    if horizon in predictions:
        pred = predictions[horizon]
        print(f"\n{horizon}:")
        print(f"  Delay Risk: {pred['delay_risk']} ({'Low' if pred['delay_risk']==0 else 'Medium' if pred['delay_risk']==1 else 'High'})")
        print(f"  Confidence: {pred['confidence']:.2%}")
        print(f"  Probabilities: Low={pred['probabilities']['low']:.2%}, Medium={pred['probabilities']['medium']:.2%}, High={pred['probabilities']['high']:.2%}")
```

### Train Models (if needed)
```bash
python scripts/multi_horizon_prediction.py
```
**Output**: Training progress and model performance metrics

---

## 4. 🌤️ **Weather Data Integration**

### View Weather Data
```python
from scripts.weather_data_integration import get_weather_for_timestamp, load_weather_data
from datetime import datetime

# Load all weather data
weather_df = load_weather_data()
print(f"Total weather records: {len(weather_df):,}")
print(f"Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")

# Get weather for specific timestamp
target_time = datetime(2023, 1, 1, 12, 0, 0)
weather = get_weather_for_timestamp(target_time, weather_df)

if weather:
    print(f"\nWeather for {target_time}:")
    print(f"  Wind Speed: {weather['wind_speed']:.1f} km/h")
    print(f"  Temperature: {weather['temperature']:.1f}°C")
    print(f"  Rainfall: {weather['rainfall']:.1f} mm")
    print(f"  Visibility: {weather['visibility']:.1f} km")
    print(f"  Wave Height: {weather['wave_height']:.2f} m")
```

### Run Weather Integration Script
```bash
python scripts/weather_data_integration.py
```
**Output**: Weather data summary and test lookup

---

## 5. 🗺️ **Real-Time Heatmaps**

### View Generated Heatmaps
```bash
# Heatmaps are saved here:
output/heatmaps/berth_heatmap_*.png
output/heatmaps/yard_heatmap_*.png
output/heatmaps/gate_heatmap_*.png
```

### Generate New Heatmaps
```bash
python scripts/heatmap_generator.py
```
**Output**: 
- Console shows generation progress
- PNG files saved in `output/heatmaps/`
- Console shows file paths and statistics

### View Heatmap Data Programmatically
```python
from scripts.heatmap_generator import generate_all_heatmaps
import pandas as pd

# Generate heatmaps
results = generate_all_heatmaps()

# View berth heatmap data
print("Berth Heatmap:")
print(f"  File: {results['berth']['file']}")
print(f"  Max Utilization: {results['berth']['max_utilization']:.2%}")
print(f"  Avg Utilization: {results['berth']['avg_utilization']:.2%}")

# View yard heatmap data
print("\nYard Heatmap:")
print(f"  File: {results['yard']['file']}")
print(f"  Max Utilization: {results['yard']['max_utilization']:.2%}")

# View gate heatmap data
print("\nGate Heatmap:")
print(f"  File: {results['gate']['file']}")
print(f"  Max Wait Time: {results['gate']['max_wait_time']:.1f} min")
print(f"  Avg Wait Time: {results['gate']['avg_wait_time']:.1f} min")
```

### Open Heatmap Images
- **Windows**: Double-click PNG files in `output/heatmaps/` folder
- **Python**: 
```python
from PIL import Image
Image.open('output/heatmaps/berth_heatmap_20251229_161621.png').show()
```

---

## 6. 🔮 **What-If Simulation**

### Run Berth Allocation Simulation
```python
from scripts.what_if_simulator import simulate_berth_allocation

# Define current state
current_state = {
    'yard_utilization_ratio': 0.85,
    'available_cranes': 20,
    'current_delay_risk': 1.5
}

# Define proposed allocation
proposed_allocation = {
    'cranes_per_berth': {1: 3, 2: 3, 3: 2},
    'berth_priority': {1: 'high', 2: 'high', 3: 'medium'}
}

# Simulate
result = simulate_berth_allocation(current_state, proposed_allocation, time_horizon_hours=24)

print("Simulation Results:")
print(f"Delay Risk: {result['current_state']['delay_risk']:.2f} -> {result['proposed_state']['delay_risk']:.2f}")
print(f"Yard Utilization: {result['current_state']['yard_utilization']:.2%} -> {result['proposed_state']['yard_utilization']:.2%}")
print(f"\nImprovements:")
print(f"  Risk Reduction: {result['improvements']['delay_risk_reduction_pct']:.1f}%")
print(f"\nCost Analysis:")
print(f"  Additional Cost: ${result['cost_analysis']['additional_crane_cost_usd']:,.2f}")
print(f"  Cost Saved: ${result['cost_analysis']['delay_cost_saved_usd']:,.2f}")
print(f"  Net Benefit: ${result['cost_analysis']['net_benefit_usd']:,.2f}")
print(f"  ROI: {result['cost_analysis']['roi_percent']:.1f}%")
print(f"\nRecommendation: {result['recommendation']}")
```

### Run Crane Deployment Simulation
```python
from scripts.what_if_simulator import simulate_crane_deployment

current_state = {
    'available_cranes': 20,
    'yard_utilization_ratio': 0.85,
    'current_delay_risk': 1.5
}

proposed_deployment = {
    'total_cranes': 25,
    'distribution': {'peak_hours': 25, 'off_hours': 15}
}

result = simulate_crane_deployment(current_state, proposed_deployment)
print(result)
```

### Compare Multiple Scenarios
```python
from scripts.what_if_simulator import compare_scenarios, simulate_berth_allocation, simulate_crane_deployment

# Run multiple scenarios
scenario1 = simulate_berth_allocation(current_state, proposed_allocation)
scenario2 = simulate_crane_deployment(current_state, proposed_deployment)

# Compare
comparison = compare_scenarios([scenario1, scenario2])
print(comparison)
```

### Run What-If Script
```bash
python scripts/what_if_simulator.py
```
**Output**: Test scenarios with detailed results

---

## 7. 📈 **View All Outputs via Streamlit Dashboard**

### Add New Features to Dashboard
The Streamlit dashboard can be enhanced to show:
- Multi-horizon predictions
- Cost & emission analysis
- Heatmaps
- What-if simulation results

### Current Dashboard
```bash
streamlit run streamlit_app/app.py
```
Then navigate to:
- **Optimization & Recommendations** page (already shows optimization results)
- Can be extended to show new features

---

## 8. 🔌 **View via FastAPI**

### Start API Server
```bash
cd api
uvicorn main:app --reload
```

### Access API Documentation
Open browser: `http://localhost:8000/docs`

### Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Predict delay risk (can be extended for multi-horizon)
curl -X POST http://localhost:8000/predict-delay-risk \
  -H "Content-Type: application/json" \
  -d '{"yard_utilization_ratio": 0.85, "crane_utilization_ratio": 0.75}'
```

---

## 9. 📊 **Quick View Commands**

### View All Output Files
```bash
# Windows PowerShell
Get-ChildItem -Recurse output\ | Where-Object {$_.Extension -in '.csv','.png','.json','.joblib'} | Select-Object FullName, Length, LastWriteTime

# Or navigate to folders:
# output/heatmaps/          - Heatmap images
# output/models/            - Trained models
# output/processed/          - Enhanced datasets
# output/optimization/       - Optimization results
```

### View Database Records
```python
from database.db_loader import load_predictions_from_db, load_optimization_run_from_db

# Load predictions
predictions = load_predictions_from_db(limit=100)
print(f"Total predictions in DB: {len(predictions)}")
print(predictions.head())

# Load optimization runs
optimization_runs = load_optimization_run_from_db(limit=10)
print(f"\nTotal optimization runs: {len(optimization_runs)}")
print(optimization_runs.head())
```

---

## 10. 📋 **Complete Example: View Everything**

```python
"""
Complete example to view all new feature outputs
"""
import pandas as pd
from pathlib import Path
from scripts.cost_emission_analysis import calculate_delay_cost
from scripts.weather_data_integration import get_weather_for_timestamp
from api.services.multi_horizon_service import predict_multi_horizon
from scripts.what_if_simulator import simulate_berth_allocation
from datetime import datetime

print("=" * 70)
print("VIEWING ALL NEW FEATURE OUTPUTS")
print("=" * 70)

# 1. Vessel Data
print("\n1. VESSEL DATA INTEGRATION")
vessel_df = pd.read_csv('output/processed/ml_features_targets_regression_refined_with_vessels.csv')
vessel_cols = [c for c in vessel_df.columns if c.startswith('vessel_')]
print(f"   Enhanced dataset: {len(vessel_df):,} rows, {len(vessel_df.columns)} columns")
print(f"   Vessel features: {len(vessel_cols)}")

# 2. Cost Analysis
print("\n2. COST & EMISSION ANALYSIS")
cost = calculate_delay_cost("VSL100000", delay_hours=2.5)
print(f"   Total Cost: ${cost['total_delay_cost_usd']:,.2f}")
print(f"   CO2 Emission: {cost['co2_emission_tonnes']:.2f} tonnes")

# 3. Multi-Horizon Predictions
print("\n3. MULTI-HORIZON PREDICTIONS")
features = {'yard_utilization_ratio': 0.85, 'crane_utilization_ratio': 0.75}
predictions = predict_multi_horizon(features)
for horizon in ['24h', '48h', '72h']:
    if horizon in predictions:
        print(f"   {horizon}: Risk={predictions[horizon]['delay_risk']}, Confidence={predictions[horizon]['confidence']:.2%}")

# 4. Weather Data
print("\n4. WEATHER DATA")
weather = get_weather_for_timestamp(datetime.now())
if weather:
    print(f"   Wind: {weather['wind_speed']:.1f} km/h, Temp: {weather['temperature']:.1f}°C")

# 5. Heatmaps
print("\n5. HEATMAPS")
heatmap_dir = Path('output/heatmaps')
if heatmap_dir.exists():
    heatmaps = list(heatmap_dir.glob('*.png'))
    print(f"   Generated heatmaps: {len(heatmaps)}")
    for hm in heatmaps[:3]:
        print(f"   - {hm.name}")

# 6. What-If Simulation
print("\n6. WHAT-IF SIMULATION")
current = {'yard_utilization_ratio': 0.85, 'available_cranes': 20, 'current_delay_risk': 1.5}
proposed = {'cranes_per_berth': {1: 3, 2: 3}}
result = simulate_berth_allocation(current, proposed)
print(f"   Net Benefit: ${result['cost_analysis']['net_benefit_usd']:,.2f}")
print(f"   Recommendation: {result['recommendation']}")

print("\n" + "=" * 70)
print("ALL OUTPUTS VIEWED SUCCESSFULLY!")
print("=" * 70)
```

---

## 📝 **Summary**

| Feature | View Method | Output Location |
|---------|-------------|-----------------|
| **Vessel Data** | Load CSV/Parquet | `output/processed/ml_features_targets_regression_refined_with_vessels.*` |
| **Cost Analysis** | Run Python script | Console output + calculations |
| **Multi-Horizon** | Use API service | `output/models/random_forest_delay_risk_*.joblib` |
| **Weather** | Load CSV or use function | `weather_data.csv` |
| **Heatmaps** | View PNG files | `output/heatmaps/*.png` |
| **What-If** | Run Python script | Console output + results dict |

---

**All outputs are ready to view! Use the examples above to explore each feature.** 🚀

