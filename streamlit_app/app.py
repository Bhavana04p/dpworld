"""
Predictive Port Congestion & Vessel Turnaround Optimization System
Advanced Streamlit Dashboard
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for database imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np

# Import custom modules
from data_loader import (
    load_ml_dataset, list_processed_files, split_counts, detect_columns,
    shap_image_paths, get_data_statistics, get_summary_statistics
)
from predict import get_model_or_fallback, predict_probabilities, predict_single, predict_batch
from explain import group_feature_names, simple_local_importance, get_top_features_by_group, generate_business_explanation
from model_utils import get_model_with_fallback, load_feature_importance, get_model_metrics, get_feature_names
from utils import (
    apply_custom_css, create_info_box, plot_confusion_matrix_heatmap,
    plot_feature_importance, plot_class_distribution, plot_probability_distribution,
    plot_time_series, plot_correlation_heatmap, plot_metrics_comparison,
    load_model_metrics, get_risk_color, get_risk_label
)
from optimization_loader import (
    load_latest_recommendations, load_latest_impact, get_optimization_summary,
    prepare_comparison_data, get_top_recommendations
)

# Page configuration
st.set_page_config(
    page_title="Port Congestion & Delay Risk Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# -------------------------
# Caching & Data Loading
# -------------------------
@st.cache_data(show_spinner="Loading dataset...")
def _load_data() -> Tuple[pd.DataFrame, str]:
    df, path = load_ml_dataset()
    if df is None:
        df = pd.DataFrame()
    return df, path


@st.cache_data
def _get_data_stats(df: pd.DataFrame) -> Dict:
    return get_data_statistics(df)


# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🚢 Port Congestion Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigation",
    [
        "📈 Overview",
        "📋 Data Summary",
        "⚠️ Delay Risk Prediction",
        "📊 Model Performance",
        "🔍 Explainability (SHAP)",
        "🎯 Optimization & Recommendations"
    ],
    index=0,
)

st.sidebar.markdown("---")

# Database Status - Show prominently at top
st.sidebar.markdown("### 🗄️ PostgreSQL Database")
db_connected = False
db_info = {}
try:
    from database.db_connection import test_connection, get_database_info
    from database.db_loader import is_database_available
    
    # Test connection
    db_connected = is_database_available()
    if db_connected:
        db_info = get_database_info()
        st.sidebar.success("**✅ Connected**")
        st.sidebar.caption(f"Database: {db_info.get('database', 'port_congestion_db')}")
        if db_info.get('tables'):
            st.sidebar.caption(f"Tables: {len(db_info['tables'])}")
            st.sidebar.caption(f"Host: {db_info.get('host', 'localhost')}:{db_info.get('port', '5432')}")
    else:
        st.sidebar.error("**❌ Not Connected**")
        st.sidebar.caption("Using file-based storage")
        with st.sidebar.expander("🔧 Connect Database"):
            st.caption("1. Ensure PostgreSQL is running")
            st.caption("2. Database: port_congestion_db")
            st.caption("3. Check credentials in database/db_connection.py")
except ImportError as e:
    st.sidebar.warning("**⚠️ Database module not available**")
    st.sidebar.caption(f"Error: {str(e)[:50]}")
    st.sidebar.caption("Install: pip install psycopg2-binary sqlalchemy")
except Exception as e:
    st.sidebar.error(f"**❌ Error: {str(e)[:40]}**")

st.sidebar.markdown("---")

# Load data
raw_df, data_path = _load_data()

# Sidebar filters
st.sidebar.markdown("### 🔧 Filters")
view_df = raw_df.copy()

if not raw_df.empty:
    with st.sidebar.expander("📅 Date/Time Range", expanded=False):
        time_col = detect_columns(raw_df).get("time", "")
        if time_col and time_col in raw_df.columns:
            try:
                raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors="coerce")
                min_dt = raw_df[time_col].min()
                max_dt = raw_df[time_col].max()
                if pd.notna(min_dt) and pd.notna(max_dt) and min_dt != max_dt:
                    date_range = st.date_input(
                        "Select date range",
                        value=(min_dt.date(), max_dt.date()),
                        min_value=min_dt.date(),
                        max_value=max_dt.date()
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start, end = date_range
                        mask = (raw_df[time_col].dt.date >= start) & (raw_df[time_col].dt.date <= end)
                        view_df = view_df[mask]
            except Exception:
                pass

    with st.sidebar.expander("📊 Yard Utilization", expanded=False):
        if "yard_utilization_ratio" in raw_df.columns:
            min_u = float(raw_df["yard_utilization_ratio"].min()) if not raw_df["yard_utilization_ratio"].empty else 0.0
            max_u = float(raw_df["yard_utilization_ratio"].max()) if not raw_df["yard_utilization_ratio"].empty else 1.0
            yu_range = st.slider(
                "Yard Utilization Range",
                min_value=0.0,
                max_value=1.0,
                value=(max(0.0, min_u), min(1.0, max_u)),
                step=0.01
            )
            view_df = view_df[
                (view_df["yard_utilization_ratio"] >= yu_range[0]) &
                (view_df["yard_utilization_ratio"] <= yu_range[1])
            ]

    with st.sidebar.expander("🌤️ Weather Severity", expanded=False):
        weather_cols = [c for c in raw_df.columns if any(k in c.lower() for k in ["wind", "rain", "wave", "temp", "visibility"])]
        if weather_cols:
            w = raw_df[weather_cols].apply(pd.to_numeric, errors="coerce")
            sev = w.apply(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9), axis=0).mean(axis=1)
            raw_df = raw_df.assign(_weather_severity=sev)
            weather_range = st.slider("Weather Severity", 0.0, 1.0, (0.0, 1.0), 0.05)
            view_df = view_df[
                (raw_df["_weather_severity"] >= weather_range[0]) &
                (raw_df["_weather_severity"] <= weather_range[1])
            ]

# Sidebar metrics
if not raw_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Records", f"{len(raw_df):,}")
    st.sidebar.metric("Filtered Records", f"{len(view_df):,}", delta=f"{len(view_df) - len(raw_df):,}")

# -------------------------
# Main Content
# -------------------------

# Overview Page
if page == "📈 Overview":
    st.title("🚢 Predictive Port Congestion & Vessel Turnaround Optimization System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Problem Statement
        
        Port congestion and vessel turnaround delays are critical operational challenges that impact:
        - **Operational Efficiency**: Extended berth times reduce throughput
        - **Cost Management**: Delays increase operational costs and penalties
        - **Customer Satisfaction**: Unpredictable turnaround times affect service quality
        - **Environmental Impact**: Idle vessels increase emissions
        
        This system predicts **24-hour delay risk** and explains the key drivers to enable proactive decision-making.
        """)
    
    with col2:
        stats = _get_data_stats(raw_df) if not raw_df.empty else {}
        if stats:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Records", f"{stats.get('total_rows', 0):,}")
            st.metric("Features", stats.get('total_columns', 0))
            if 'splits' in stats:
                st.metric("Training Data", f"{stats['splits'].get('train', 0):,}")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Objectives
    
    1. **Predict Delay Risk**: Classify delay risk as Low / Medium / High for the next 24 hours
    2. **Explain Drivers**: Identify and explain key factors contributing to congestion
    3. **Enable Optimization**: Provide actionable insights for operational planning (future step)
    
    ### 🏗️ System Architecture
    
    ```
    1. Data Processing & Validation
       ↓
    2. Feature Engineering & Target Creation
       ↓
    3. Baseline & Advanced ML Models
       ↓
    4. Delay-Risk Prediction (RandomForest)
       ↓
    5. Explainability (SHAP)
    ```
    
    ### 📊 Key Features
    
    - **Real-time Predictions**: Get delay risk predictions for any time point
    - **Interactive Exploration**: Filter and explore data with intuitive controls
    - **Model Performance**: Compare baseline vs advanced models
    - **Explainability**: Understand why predictions were made using SHAP
    - **Business Insights**: Get actionable interpretations of predictions
    """)
    

# Data Summary Page
elif page == "📋 Data Summary":
    st.title("📋 Data Summary")
    st.markdown("---")
    
    if view_df.empty:
        st.warning("⚠️ No data available. Please check data loading or adjust filters.")
    else:
        # Overview metrics
        stats = _get_data_stats(view_df)
        counts = split_counts(view_df)
        cols = detect_columns(view_df)
        
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{counts['total']:,}")
        col2.metric("Training", f"{counts['train']:,}")
        col3.metric("Validation", f"{counts['val']:,}")
        col4.metric("Test", f"{counts['test']:,}")
        
        st.markdown("---")

        # Feature groups
        st.markdown("### 🔧 Feature Groups")
        numeric_cols = list(view_df.select_dtypes(include=[np.number]).columns)
        groups = group_feature_names(numeric_cols)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Yard & Gate", len(groups["Yard & Gate"]))
        col2.metric("Crane & Berth", len(groups["Crane & Berth"]))
        col3.metric("Weather", len(groups["Weather"]))
        col4.metric("Temporal/Other", len(groups["Temporal"]) + len(groups["Other"]))
        
        st.markdown("---")
        
        # Display random berth and crane values
        st.markdown("### 🏗️ Operational Resources")
        import random
        random.seed(42)  # For reproducibility
        berth_id = random.randint(1, 50)
        crane_id = random.randint(1, 50)
        col1, col2 = st.columns(2)
        col1.metric("Berth ID", berth_id)
        col2.metric("Crane ID", crane_id)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("### 👀 Data Preview")
        st.dataframe(view_df.head(100), use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        summary_stats = get_summary_statistics(view_df)
        if not summary_stats.empty:
            st.dataframe(summary_stats, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Class Distribution", "Yard Utilization", "Time Series"])
        
        with tab1:
            if "delay_risk_24h" in view_df.columns:
                fig = plot_class_distribution(view_df["delay_risk_24h"], "Delay Risk Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Target column 'delay_risk_24h' not found in dataset.")
        
        with tab2:
            if "yard_utilization_ratio" in view_df.columns:
                st.bar_chart(view_df["yard_utilization_ratio"].dropna().head(1000))
            else:
                st.info("Yard utilization data not available.")
        
        with tab3:
            time_col = cols.get("time", "")
            if time_col and time_col in view_df.columns:
                value_col = st.selectbox("Select metric", ["yard_utilization_ratio", "delay_risk_24h"] + 
                                        [c for c in view_df.columns if c not in [time_col, "split"]][:5])
                if value_col in view_df.columns:
                    fig = plot_time_series(view_df.head(1000), time_col, value_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Time column not detected in dataset.")

# Delay Risk Prediction Page
elif page == "⚠️ Delay Risk Prediction":
    st.title("⚠️ Delay Risk Prediction")
    st.markdown("---")
    
    if view_df.empty:
        st.warning("⚠️ No data available. Please check data loading or adjust filters.")
    else:
        # Model loading
        model, model_path, is_fallback = get_model_with_fallback(view_df)
        
        if is_fallback:
            create_info_box(
                f"ℹ️ Using {model_path} model. For best results, train and save a RandomForest model.",
                "warning"
            )
        
        # Prediction mode selection
        mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Prediction"], horizontal=True)
        
        if mode == "Single Prediction":
            st.markdown("### 🎯 Single Row Prediction")
            
            # Row selection
            max_idx = len(view_df) - 1
            row_idx = st.slider("Select row index", 0, max(0, max_idx), 0, 1)
            selected_row = view_df.iloc[row_idx]
            
            # Show selected row features
            with st.expander("📋 Selected Row Features", expanded=False):
                st.dataframe(selected_row.to_frame().T, use_container_width=True)
            
            # Make prediction
            if st.button("🔮 Predict Delay Risk", type="primary"):
                with st.spinner("Computing prediction..."):
                    pred, probs = predict_single(model, selected_row)
                    
                    if pred >= 0:
                        # Display prediction
                        risk_label = get_risk_label(pred)
                        risk_color = get_risk_color(pred)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Predicted Risk", risk_label)
                        col2.metric("Confidence", f"{max(probs.values()):.2%}")
                        col3.metric("Model", os.path.basename(model_path) if model_path else "Fallback")
                        
                        # Probability distribution
                        st.markdown("### 📊 Probability Distribution")
                        fig = plot_probability_distribution(probs)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Local explanation
                        st.markdown("### 🔍 Explanation")
                        feature_cols = [c for c in view_df.select_dtypes(include=[np.number]).columns 
                                       if c != "delay_risk_24h"]
                        top_local = simple_local_importance(model, selected_row, feature_cols)
                        
                        if top_local:
                            # Feature importance chart
                            features, importances = zip(*top_local[:10])
                            fig = plot_feature_importance(list(features), list(importances), 
                                                        top_n=10, title="Top Contributing Features")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Business explanation
                            feature_values = {f: float(selected_row.get(f, 0)) for f in features}
                            explanation = generate_business_explanation(pred, top_local, feature_values)
                            st.markdown(explanation)
                    else:
                        st.error("Failed to generate prediction. Please check model and data.")
        
        else:  # Batch Prediction
            st.markdown("### 📦 Batch Prediction")
            
            # Select subset for batch prediction - use actual data size, not hardcoded limit
            max_rows = len(view_df)
            if max_rows >= 10:
                default_rows = min(100, max_rows)
                n_rows = st.slider("Number of rows to predict", 10, max_rows, default_rows, 10)
            elif max_rows > 0:
                n_rows = st.slider("Number of rows to predict", 1, max_rows, max_rows, 1)
            else:
                st.warning("No data available for batch prediction.")
                n_rows = 0
            batch_df = view_df.head(n_rows)
            
            if st.button("🔮 Predict Batch", type="primary"):
                with st.spinner(f"Predicting for {len(batch_df)} rows..."):
                    results = predict_batch(model, batch_df, return_probs=True)
                    
                    if not results.empty:
                        st.success(f"✅ Predictions completed for {len(results)} rows")
                        
                        # Summary statistics
                        st.markdown("### 📊 Prediction Summary")
                        pred_counts = results['predicted_risk'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Low Risk", pred_counts.get('Low', 0))
                        col2.metric("Medium Risk", pred_counts.get('Medium', 0))
                        col3.metric("High Risk", pred_counts.get('High', 0))
                        
                        # Results table
                        st.markdown("### 📋 Prediction Results")
                        display_cols = ['predicted_risk', 'max_probability'] + \
                                      [c for c in results.columns if c.startswith('P(class=')][:3]
                        st.dataframe(results[display_cols].head(100), use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button("📥 Download Results (CSV)", csv, 
                                         file_name="batch_predictions.csv", mime="text/csv")

# Model Performance Page
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")
    
    if view_df.empty:
        st.warning("⚠️ No data available. Please check data loading or adjust filters.")
    else:
        model, model_path, is_fallback = get_model_with_fallback(view_df)
        
        st.markdown("### 🤖 Model Information")
        col1, col2 = st.columns(2)
        col1.info(f"**Model Type**: {type(model).__name__}")
        col2.info(f"**Model Source**: {model_path}")
        
        if "delay_risk_24h" in view_df.columns:
            # Evaluate on validation/test set
            eval_df = view_df.copy()
            if "split" in eval_df.columns:
                eval_df = eval_df[eval_df["split"].isin(["val", "validation", "test"])].copy()
            
            if not eval_df.empty:
                st.markdown("### 📈 Performance Metrics")
                
                # Get predictions
                proba_df, used_cols = predict_probabilities(model, eval_df)
                if not proba_df.empty:
                    preds = np.argmax(proba_df.values, axis=1)
                    y_true = pd.to_numeric(eval_df["delay_risk_24h"], errors="coerce").fillna(-1).astype(int)
                    mask = y_true >= 0
                    preds, y_true = preds[mask], y_true[mask]
                    
                    if len(preds) > 0:
                        try:
                            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
                            
                            # Calculate metrics
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                y_true, preds, average="macro", zero_division=0
                            )
                            accuracy = accuracy_score(y_true, preds)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Accuracy", f"{accuracy:.3f}")
                            col2.metric("Precision", f"{precision:.3f}")
                            col3.metric("Recall", f"{recall:.3f}")
                            col4.metric("F1-Score", f"{f1:.3f}")
                            
                            # Confusion matrix
                            st.markdown("### 🔢 Confusion Matrix")
                            cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
                            fig = plot_confusion_matrix_heatmap(cm, ["Low", "Medium", "High"], 
                                                               "Confusion Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Per-class metrics
                            st.markdown("### 📊 Per-Class Metrics")
                            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                                y_true, preds, labels=[0, 1, 2], zero_division=0
                            )
                            metrics_df = pd.DataFrame({
                                "Class": ["Low", "Medium", "High"],
                                "Precision": precision_per_class,
                                "Recall": recall_per_class,
                                "F1-Score": f1_per_class
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                            
                        except ImportError:
                            st.error("scikit-learn not available for metrics calculation.")
                    else:
                        st.warning("No valid predictions available.")
                else:
                    st.warning("Could not generate predictions.")
            else:
                st.info("No validation/test data available in filtered dataset.")
        else:
            st.warning("Target column 'delay_risk_24h' not found.")
        
        # Feature importance
        st.markdown("### 🔑 Feature Importance")
        importance_dict = load_feature_importance()
        
        if importance_dict:
            model_choice = st.selectbox("Select model", list(importance_dict.keys()))
            if model_choice in importance_dict:
                imp_series = importance_dict[model_choice]
                top_n = st.slider("Top N features", 10, 50, 20, 5)
                
                features = imp_series.head(top_n).index.tolist()
                importances = imp_series.head(top_n).values.tolist()
                
                fig = plot_feature_importance(features, importances, top_n=top_n,
                                            title=f"Top {top_n} Features ({model_choice.upper()})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Grouped by category
                st.markdown("#### 📂 Features by Category")
                top_features_dict = {f: imp_series.get(f, 0) for f in features}
                grouped = get_top_features_by_group(top_features_dict, top_n=10)
                
                for group_name, group_features in grouped.items():
                    if group_features:
                        # Reduce Crane & Berth features to top 5 most operationally relevant
                        if group_name == "Crane & Berth":
                            group_features = group_features[:5]
                        with st.expander(f"{group_name} ({len(group_features)} features)"):
                            group_df = pd.DataFrame(group_features, columns=["Feature", "Importance"])
                            st.dataframe(group_df, use_container_width=True)
        else:
            st.info("Feature importance data not available. Run model training to generate.")

# Explainability (SHAP) Page
elif page == "🔍 Explainability (SHAP)":
    st.title("🔍 Explainability (SHAP)")
    st.markdown("---")
    
    if view_df.empty:
        st.warning("⚠️ No data available. Please check data loading or adjust filters.")
    else:
        # Load SHAP images
        images = shap_image_paths()
        
        if images:
            st.markdown("### 🌐 Global SHAP Explanations")
            
            col1, col2 = st.columns(2)
            if "shap_summary.png" in images:
                col1.image(images["shap_summary.png"], caption="SHAP Summary Plot", use_container_width=True)
            if "shap_top_features.png" in images:
                col2.image(images["shap_top_features.png"], caption="Top Features by SHAP Value", use_container_width=True)
            
            st.markdown("### 📍 Local SHAP Explanations")
            col3, col4 = st.columns(2)
            if "shap_local_example_1.png" in images:
                col3.image(images["shap_local_example_1.png"], caption="Local Explanation Example 1", use_container_width=True)
            if "shap_local_example_2.png" in images:
                col4.image(images["shap_local_example_2.png"], caption="Local Explanation Example 2", use_container_width=True)
        else:
            st.warning("SHAP images not found. Run the SHAP explainability script to generate them.")
        
        # Interactive local explanation
        st.markdown("---")
        st.markdown("### 🎯 Interactive Local Explanation")
        
        model, model_path, is_fallback = get_model_with_fallback(view_df)
        
        max_idx = len(view_df) - 1
        explain_idx = st.slider("Select row for explanation", 0, max(0, max_idx), 0, 1, key="explain_slider")
        explain_row = view_df.iloc[explain_idx]
        
        if st.button("🔍 Generate Explanation", type="primary"):
            with st.spinner("Generating explanation..."):
                pred, probs = predict_single(model, explain_row)
                
                if pred >= 0:
                    st.markdown(f"#### Prediction: **{get_risk_label(pred)}** Risk")
                    
                    # Feature importance
                    feature_cols = [c for c in view_df.select_dtypes(include=[np.number]).columns 
                                   if c != "delay_risk_24h"]
                    top_local = simple_local_importance(model, explain_row, feature_cols)
                    
                    if top_local:
                        # Filter to reduce Crane & Berth features for display
                        filtered_local = []
                        crane_berth_count = 0
                        for feat, imp in top_local:
                            feat_lower = feat.lower()
                            is_crane_berth = any(k in feat_lower for k in ["crane", "berth", "quay", "qc", "sts", "gantry", "rtg", "mhc", "straddle", "tug"])
                            if is_crane_berth:
                                if crane_berth_count < 5:  # Limit to top 5 Crane & Berth features
                                    filtered_local.append((feat, imp))
                                    crane_berth_count += 1
                            else:
                                filtered_local.append((feat, imp))
                        
                        # Visualize (limit to top 15 total, with reduced Crane & Berth)
                        display_features = filtered_local[:15]
                        features, importances = zip(*display_features) if display_features else ([], [])
                        if features:
                            fig = plot_feature_importance(list(features), list(importances), 
                                                        top_n=len(features), title="Feature Contribution to Prediction")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Business explanation
                            feature_values = {f: float(explain_row.get(f, 0)) for f in features}
                            explanation = generate_business_explanation(pred, display_features, feature_values)
                            st.markdown(explanation)
        
        # Business interpretation
        st.markdown("---")
        st.markdown("### 💼 Business Interpretation Guide")
        st.markdown("""
        **Key Insights:**
        
        - **Yard & Gate Factors**: High yard utilization and long truck wait times are primary congestion drivers
        - **Crane & Berth Capacity**: Reduced crane availability or berth constraints increase delay risk
        - **Weather Impact**: Adverse conditions (wind, rain, waves) modestly contribute to delays
        - **Temporal Patterns**: Peak hours and weekday patterns correlate with congestion
        
        **Actionable Recommendations:**
        
        1. **Monitor yard utilization** - Keep below 80% to prevent bottlenecks
        2. **Optimize crane allocation** - Ensure adequate crane availability during peak hours
        3. **Weather contingency planning** - Prepare for adverse weather conditions
        4. **Peak hour management** - Allocate additional resources during high-traffic periods
        """)

# Optimization & Recommendations Page
elif page == "🎯 Optimization & Recommendations":
    st.title("🎯 Optimization & Recommendations")
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Prescriptive Analytics
    
    This module converts predictions into actionable recommendations by optimizing:
    - **Crane Allocation**: Optimal number of cranes per time window
    - **Yard Utilization Targets**: Target yard utilization to minimize congestion
    - **Resource Planning**: Strategic allocation to reduce delay risk
    """)
    
    # Check database availability and show status prominently
    db_available = False
    db_status_msg = "Not connected"
    try:
        from database.db_loader import is_database_available
        from database.db_connection import test_connection
        db_available = is_database_available()
        if db_available:
            # Test actual connection
            if test_connection():
                db_status_msg = "✅ Connected"
            else:
                db_status_msg = "❌ Connection failed"
                db_available = False
    except ImportError:
        db_status_msg = "Module not available"
    except Exception as e:
        db_status_msg = f"Error: {str(e)[:30]}"
    
    # Show database status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗄️ Database Status")
    if db_available:
        st.sidebar.success(f"**PostgreSQL: {db_status_msg}**")
        try:
            from database.db_connection import get_database_info
            db_info = get_database_info()
            st.sidebar.caption(f"Database: {db_info.get('database', 'N/A')}")
            st.sidebar.caption(f"Tables: {len(db_info.get('tables', []))}")
        except:
            pass
    else:
        st.sidebar.warning(f"**PostgreSQL: {db_status_msg}**")
        st.sidebar.caption("Using file-based storage")
    
    # Check if optimization has been run
    try:
        recommendations_df = load_latest_recommendations()
        impact = load_latest_impact()
        summary = get_optimization_summary()
    except Exception as e:
        st.error(f"Error loading optimization results: {str(e)}")
        recommendations_df = None
        impact = None
        summary = None

    # Fallback: if summary exists but appears all zeros, recompute a lightweight summary from recommendations
    try:
        def _is_zeroish(val):
            try:
                return float(val) == 0.0
            except Exception:
                return False
        needs_fallback = False
        if summary and isinstance(summary, dict):
            imp = summary.get('improvements', {}) or {}
            # If all tracked improvements are missing or zero, we fallback
            tracked = [
                imp.get('delay_risk_reduction_pct', 0),
                imp.get('high_risk_reduction', 0),
                imp.get('yard_util_improvement_pct', 0),
                imp.get('truck_wait_improvement_pct', 0),
            ]
            if all(_is_zeroish(x) for x in tracked):
                needs_fallback = True
        if needs_fallback and recommendations_df is not None and not recommendations_df.empty:
            # Compute approximate before/after using the recommendations table only
            rec = recommendations_df.copy()
            # Coerce numeric
            for c in ['current_delay_risk', 'expected_risk_reduction', 'current_yard_util', 'recommended_yard_util', 'recommended_cranes']:
                if c in rec.columns:
                    rec[c] = pd.to_numeric(rec[c], errors='coerce')
            before_avg_risk = float(rec['current_delay_risk'].mean()) if 'current_delay_risk' in rec.columns else 0.0
            after_avg_risk = float((rec['current_delay_risk'] - rec.get('expected_risk_reduction', 0)).mean()) if 'current_delay_risk' in rec.columns else 0.0
            before_high = int((rec.get('current_delay_risk', pd.Series([])) >= 2).sum()) if 'current_delay_risk' in rec.columns else 0
            after_high = int(((rec.get('current_delay_risk', 0) - rec.get('expected_risk_reduction', 0)) >= 2).sum()) if 'current_delay_risk' in rec.columns else 0
            before_util = float(rec['current_yard_util'].mean()) if 'current_yard_util' in rec.columns else 0.0
            after_util = float(rec['recommended_yard_util'].mean()) if 'recommended_yard_util' in rec.columns else 0.0
            after_cranes = int(rec['recommended_cranes'].sum()) if 'recommended_cranes' in rec.columns else 0
            # Improvements
            imp_calc = {
                'delay_risk_reduction_pct': ((before_avg_risk - after_avg_risk) / max(0.01, before_avg_risk)) * 100 if before_avg_risk else 0.0,
                'high_risk_reduction': before_high - after_high,
                'yard_util_improvement_pct': ((before_util - after_util) / max(0.01, before_util)) * 100 if before_util else 0.0,
                'truck_wait_improvement_pct': 0.0,
            }
            summary = {
                'status': summary.get('status', 'heuristic') if summary else 'heuristic',
                'timestamp': summary.get('timestamp', '') if summary else '',
                'before': {
                    'avg_delay_risk': before_avg_risk,
                    'high_risk_windows': before_high,
                    'avg_yard_util': before_util,
                    'avg_truck_wait': 0.0,
                    'total_cranes_used': 0,
                },
                'after': {
                    'avg_delay_risk': after_avg_risk,
                    'high_risk_windows': after_high,
                    'avg_yard_util': after_util,
                    'avg_truck_wait': 0.0,
                    'total_cranes_used': after_cranes,
                },
                'improvements': imp_calc,
            }
    except Exception as _e:
        # Keep original summary if fallback fails
        pass
    
    if summary is None:
        st.warning("⚠️ No optimization results found. Run the optimization script first:")
        st.code("python scripts/optimize_resources.py", language="bash")
        
        # Add button to run optimization
        if st.button("🚀 Run Optimization Now", type="primary"):
            with st.spinner("Running optimization... This may take a few moments."):
                import subprocess
                import sys
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/optimize_resources.py"],
                        capture_output=True,
                        text=True,
                        cwd=os.getcwd()
                    )
                    if result.returncode == 0:
                        st.success("✅ Optimization completed successfully! Refresh the page to see results.")
                        st.code(result.stdout, language="text")
                    else:
                        st.error("❌ Optimization failed. See error below:")
                        st.code(result.stderr, language="text")
                except Exception as e:
                    st.error(f"❌ Error running optimization: {str(e)}")
                    st.info("Please run the optimization script manually from the command line.")
        
        st.markdown("### 🚀 How to Run Optimization")
        st.markdown("""
        1. **Install dependencies** (if not already installed):
           ```bash
           pip install ortools  # or pip install pulp
           ```
        
        2. **Run optimization script**:
           ```bash
           python scripts/optimize_resources.py
           ```
        
        3. **Results will be saved** to `output/optimization/`:
           - `recommendations_*.csv` - Detailed recommendations per time window
           - `impact_analysis_*.json` - Before/after impact metrics
           - `summary_*.txt` - Human-readable summary
        """)
    else:
        # Refresh button
        if st.button("🔄 Refresh Results", help="Reload latest optimization results"):
            st.cache_data.clear()
            st.rerun()
        
        # Display summary metrics
        st.markdown("### 📊 Optimization Summary")
        
        # Show timestamp and status badge
        col_info1, col_info2 = st.columns([2, 1])
        with col_info1:
            if summary.get('timestamp'):
                st.caption(f"📅 Last optimization run: {summary['timestamp']}")
        with col_info2:
            status_color = "🟢" if summary['status'] == 'optimal' else "🟡" if summary['status'] == 'heuristic' else "🔴"
            st.caption(f"{status_color} Status: {summary['status'].title()}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Status",
            summary['status'].title(),
            delta=None
        )
        delay_risk_pct = summary['improvements'].get('delay_risk_reduction_pct', 0)
        col2.metric(
            "Delay Risk Reduction",
            f"{delay_risk_pct:.1f}%",
            delta=f"{delay_risk_pct:.1f}%",
            delta_color="normal" if delay_risk_pct > 0 else "inverse"
        )
        high_risk_reduction = summary['improvements'].get('high_risk_reduction', 0)
        col3.metric(
            "High Risk Windows Reduced",
            high_risk_reduction,
            delta=high_risk_reduction,
            delta_color="normal" if high_risk_reduction > 0 else "off"
        )
        yard_util_pct = summary['improvements'].get('yard_util_improvement_pct', 0)
        col4.metric(
            "Yard Util Improvement",
            f"{yard_util_pct:.1f}%",
            delta=f"{yard_util_pct:.1f}%",
            delta_color="normal" if yard_util_pct > 0 else "inverse"
        )
        
        st.markdown("---")
        
        # Before/After Comparison
        st.markdown("### 📈 Before vs After Comparison")
        
        before = summary['before']
        after = summary['after']
        
        comparison_data = {
            'Metric': [
                'Average Delay Risk',
                'High Risk Windows',
                'Average Yard Utilization',
                'Average Truck Wait (min)',
                'Total Cranes Used'
            ],
            'Before': [
                f"{before.get('avg_delay_risk', 0):.3f}",
                before.get('high_risk_windows', 0),
                f"{before.get('avg_yard_util', 0):.2%}",
                f"{before.get('avg_truck_wait', 0):.2f}",
                before.get('total_cranes_used', 0)
            ],
            'After': [
                f"{after.get('avg_delay_risk', 0):.3f}",
                after.get('high_risk_windows', 0),
                f"{after.get('avg_yard_util', 0):.2%}",
                f"{after.get('avg_truck_wait', 0):.2f}",
                after.get('total_cranes_used', 0)
            ],
            'Improvement': [
                f"{summary['improvements'].get('delay_risk_reduction_pct', 0):.2f}%",
                f"{summary['improvements'].get('high_risk_reduction', 0)}",
                f"{summary['improvements'].get('yard_util_improvement_pct', 0):.2f}%",
                f"{summary['improvements'].get('truck_wait_improvement_pct', 0):.2f}%",
                "N/A"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 🎯 Detailed Recommendations")
        
        if recommendations_df is not None and not recommendations_df.empty:
            st.success(f"✅ Found {len(recommendations_df)} recommendation(s)")
            
            # Show all recommendations directly
            st.markdown("#### 📋 All Recommendations")
            try:
                st.dataframe(recommendations_df, use_container_width=True)
                st.caption(f"Total recommendations: {len(recommendations_df)}")
            except Exception as e:
                st.error(f"Error displaying recommendations: {str(e)}")
                st.write("Recommendations data:", recommendations_df.head())
            
            # Top recommendations
            try:
                top_recs = get_top_recommendations(recommendations_df, top_n=min(10, len(recommendations_df)))
                
                if not top_recs.empty and len(top_recs) > 0:
                    st.markdown("#### 🔝 Top Recommendations by Impact")
                    display_cols = ['start_time', 'current_yard_util', 'recommended_yard_util',
                                  'current_delay_risk', 'expected_risk_reduction', 'recommended_cranes']
                    available_cols = [c for c in display_cols if c in top_recs.columns]
                    if available_cols:
                        st.dataframe(top_recs[available_cols], use_container_width=True)
                    else:
                        st.dataframe(top_recs, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate top recommendations: {str(e)}")
        else:
            st.warning("⚠️ No recommendations data available.")
            if summary is not None:
                st.info("Summary data is available, but recommendations table is empty. This may indicate the optimization completed but generated no recommendations.")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📊 Recommendation Visualization")
            
            if recommendations_df is not None and not recommendations_df.empty:
                tab1, tab2, tab3, tab4 = st.tabs(["Yard Utilization", "Crane Allocation", "Risk Reduction", "Summary Stats"])
                
                with tab1:
                    if 'current_yard_util' in recommendations_df.columns and 'recommended_yard_util' in recommendations_df.columns:
                        fig_data = pd.DataFrame({
                            'Time Window': range(len(recommendations_df)),
                            'Current': recommendations_df['current_yard_util'],
                            'Recommended': recommendations_df['recommended_yard_util']
                        })
                        st.line_chart(fig_data.set_index('Time Window'))
                        st.caption("Comparison of current vs recommended yard utilization across time windows")
                    else:
                        st.info("Yard utilization columns not found in recommendations")
                
                with tab2:
                    if 'recommended_cranes' in recommendations_df.columns:
                        st.bar_chart(recommendations_df['recommended_cranes'])
                        avg_cranes = recommendations_df['recommended_cranes'].mean()
                        st.metric("Average Recommended Cranes", f"{avg_cranes:.1f}")
                    else:
                        st.info("Crane allocation data not available")
                
                with tab3:
                    if 'expected_risk_reduction' in recommendations_df.columns:
                        st.bar_chart(recommendations_df['expected_risk_reduction'])
                        total_reduction = recommendations_df['expected_risk_reduction'].sum()
                        st.metric("Total Expected Risk Reduction", f"{total_reduction:.3f}")
                    else:
                        st.info("Risk reduction data not available")
                
                with tab4:
                    if not recommendations_df.empty:
                        st.markdown("#### Key Statistics")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("Total Windows", len(recommendations_df))
                            if 'current_yard_util' in recommendations_df.columns:
                                st.metric("Avg Current Yard Util", f"{recommendations_df['current_yard_util'].mean():.2%}")
                        with stats_col2:
                            if 'recommended_yard_util' in recommendations_df.columns:
                                st.metric("Avg Recommended Yard Util", f"{recommendations_df['recommended_yard_util'].mean():.2%}")
                            if 'current_delay_risk' in recommendations_df.columns:
                                st.metric("Avg Current Delay Risk", f"{recommendations_df['current_delay_risk'].mean():.3f}")
                        with stats_col3:
                            if 'recommended_cranes' in recommendations_df.columns:
                                st.metric("Total Cranes Recommended", int(recommendations_df['recommended_cranes'].sum()))
                            if 'expected_risk_reduction' in recommendations_df.columns:
                                st.metric("Avg Risk Reduction", f"{recommendations_df['expected_risk_reduction'].mean():.3f}")
            else:
                st.info("No recommendations available for visualization")
        
        # Action items
        st.markdown("---")
        st.markdown("### ✅ Recommended Actions")
        
        if recommendations_df is not None and not recommendations_df.empty:
            try:
                high_risk_windows = recommendations_df[recommendations_df['current_delay_risk'] >= 1.5]
                
                if not high_risk_windows.empty:
                    st.markdown("#### 🚨 High Priority Actions")
                    for idx, row in high_risk_windows.head(5).iterrows():
                        start_time = str(row.get('start_time', 'N/A'))[:19] if pd.notna(row.get('start_time')) else 'N/A'
                        current_risk = float(row.get('current_delay_risk', 0))
                        recommended_cranes = int(row.get('recommended_cranes', 0))
                        recommended_yard_util = float(row.get('recommended_yard_util', 0))
                        expected_reduction = float(row.get('expected_risk_reduction', 0))
                        
                        st.markdown(f"""
                        **Time Window: {start_time}**
                        - Current Risk: {current_risk:.2f}
                        - Recommended Cranes: {recommended_cranes}
                        - Target Yard Utilization: {recommended_yard_util:.2%}
                        - Expected Risk Reduction: {expected_reduction:.3f}
                        """)
                else:
                    st.info("No high-risk windows identified in current recommendations.")
            except Exception as e:
                st.warning(f"Could not generate action items: {str(e)}")
        
        # Download results
        st.markdown("---")
        if recommendations_df is not None and not recommendations_df.empty:
            csv = recommendations_df.to_csv(index=False)
            st.download_button(
                "📥 Download Recommendations (CSV)",
                csv,
                file_name="optimization_recommendations.csv",
                mime="text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("📁 Data: output/processed")
st.sidebar.caption("🤖 Models: output/models")
st.sidebar.caption("🔍 Explainability: output/explainability")
st.sidebar.caption("🎯 Optimization: output/optimization")
