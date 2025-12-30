# Port Congestion Dashboard - Quick Start Guide

## 🚀 Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📋 Features

### 1. Overview Page
- Project introduction and objectives
- System architecture flow
- Quick statistics

### 2. Data Summary Page
- Dataset overview and statistics
- Feature group breakdown
- Interactive visualizations
- Data preview and summary statistics

### 3. Delay Risk Prediction Page
- **Single Prediction**: Predict delay risk for a specific row
- **Batch Prediction**: Predict for multiple rows at once
- Interactive explanations with feature importance
- Business-friendly interpretations

### 4. Model Performance Page
- Model metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Per-class performance metrics
- Feature importance visualization
- Features grouped by category

### 5. Explainability (SHAP) Page
- Global SHAP visualizations
- Local explanation examples
- Interactive local explanations
- Business interpretation guide

## 🔧 Sidebar Filters

- **Date/Time Range**: Filter data by time period
- **Yard Utilization**: Filter by yard utilization ratio
- **Weather Severity**: Filter by weather conditions

## 📊 Data Requirements

The dashboard expects:
- Processed ML dataset at: `output/processed/ml_features_targets_regression_refined.parquet` (or CSV)
- Trained models at: `output/models/` (optional, will use fallback if not found)
- SHAP images at: `output/explainability/` (optional)

## 💡 Tips

1. **Model Loading**: If no saved model is found, the app will use a heuristic fallback or train a quick model
2. **Filtering**: Use sidebar filters to focus on specific time periods or conditions
3. **Batch Predictions**: Limited to 1000 rows for performance
4. **Visualizations**: All charts are interactive - hover for details, zoom, pan

## 🎨 UI Features

- Professional, enterprise-style design
- Responsive layout
- Interactive Plotly charts
- Color-coded risk levels (Green=Low, Orange=Medium, Red=High)
- Clean, minimal aesthetic

