# Viewing Optimization Results in Streamlit Dashboard

## Quick Start

1. **Start the Streamlit dashboard**:
   ```bash
   streamlit run streamlit_app/app.py
   ```

2. **Navigate to Optimization Page**:
   - In the sidebar, click on **"🎯 Optimization & Recommendations"**

3. **View Results**:
   - The page will automatically load the latest optimization results
   - If no results are found, you'll see instructions to run optimization first

## What You'll See

### 1. Optimization Summary (Top Section)
- **Status**: Optimization status (Optimal/Heuristic/Infeasible)
- **Delay Risk Reduction**: Percentage improvement in delay risk
- **High Risk Windows Reduced**: Number of high-risk windows eliminated
- **Yard Util Improvement**: Percentage improvement in yard utilization
- **Last Run Timestamp**: When optimization was last executed

### 2. Before vs After Comparison Table
Side-by-side comparison showing:
- Average Delay Risk (before → after)
- High Risk Windows (before → after)
- Average Yard Utilization (before → after)
- Average Truck Wait Time (before → after)
- Total Cranes Used (after optimization)

### 3. Detailed Recommendations
- **Top 10 Recommendations**: Highest impact recommendations sorted by risk reduction
- **All Recommendations**: Complete list in expandable section
- Shows for each time window:
  - Start/End Time
  - Current vs Recommended Yard Utilization
  - Current Delay Risk
  - Expected Risk Reduction
  - Recommended Number of Cranes

### 4. Visualizations (Interactive Charts)
- **Yard Utilization Tab**: Line chart comparing current vs recommended utilization
- **Crane Allocation Tab**: Bar chart showing recommended cranes per window
- **Risk Reduction Tab**: Bar chart showing expected risk reduction per window
- **Summary Stats Tab**: Key statistics and metrics

### 5. Recommended Actions
- **High Priority Actions**: Specific recommendations for high-risk windows
- Shows actionable items with:
  - Time window
  - Current risk level
  - Recommended crane allocation
  - Target yard utilization
  - Expected impact

### 6. Download Options
- **Download Recommendations (CSV)**: Export all recommendations to CSV file

## Features

### Refresh Results
- Click **"🔄 Refresh Results"** button to reload latest optimization results
- Useful after running a new optimization

### Run Optimization from Dashboard
- If no results exist, click **"🚀 Run Optimization Now"** button
- The optimization will run in the background
- Results will be displayed after completion

## Troubleshooting

### "No optimization results found"
**Solution**: 
1. Run optimization script: `python scripts/optimize_resources.py`
2. Or click "🚀 Run Optimization Now" button in dashboard
3. Then click "🔄 Refresh Results"

### Results not updating
**Solution**: 
1. Click "🔄 Refresh Results" button
2. Or refresh the entire page (F5)

### Charts not showing
**Solution**: 
1. Ensure recommendations CSV file exists in `output/optimization/`
2. Check that CSV has required columns
3. Try refreshing the page

## Example Output

When optimization results are available, you'll see:

```
📊 Optimization Summary
Status: Optimal | Delay Risk Reduction: -14.2% | High Risk Windows Reduced: 0 | Yard Util Improvement: -12.8%

📈 Before vs After Comparison
[Table showing metrics comparison]

🎯 Detailed Recommendations
Top 10 Recommendations by Impact
[Table with top recommendations]

📊 Recommendation Visualization
[Interactive charts in tabs]

✅ Recommended Actions
🚨 High Priority Actions
[Action items for high-risk windows]
```

## Tips

1. **Sort Recommendations**: The top 10 are automatically sorted by impact (risk reduction)
2. **Filter High-Risk**: Look at "High Priority Actions" section for urgent recommendations
3. **Compare Metrics**: Use "Before vs After" table to see overall impact
4. **Visual Analysis**: Use charts to identify patterns across time windows
5. **Export Data**: Download CSV for further analysis in Excel/Python

