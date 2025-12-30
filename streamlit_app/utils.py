"""
Utility functions for the Streamlit dashboard
"""
import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def apply_custom_css():
    """Apply professional custom CSS styling"""
    st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        margin-top: 1.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Cards */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.25rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1565c0;
    }
    </style>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, delta: Optional[str] = None, delta_color: str = "normal"):
    """Create a styled metric card"""
    if delta:
        st.metric(title, value, delta=delta, delta_color=delta_color)
    else:
        st.metric(title, value)


def create_info_box(message: str, type: str = "info"):
    """Create styled info/warning/success boxes"""
    colors = {
        "info": ("#e3f2fd", "#2196f3"),
        "success": ("#e8f5e9", "#4caf50"),
        "warning": ("#fff3e0", "#ff9800"),
        "error": ("#ffebee", "#f44336")
    }
    bg, border = colors.get(type, colors["info"])
    st.markdown(
        f'<div style="background-color: {bg}; border-left: 4px solid {border}; padding: 1rem; margin: 1rem 0; border-radius: 0.25rem;">{message}</div>',
        unsafe_allow_html=True
    )


def plot_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix"):
    """Create an interactive confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {i}" for i in labels],
        y=[f"True {i}" for i in labels],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=500
    )
    return fig


def plot_feature_importance(feature_names: List[str], importances: List[float], top_n: int = 20, title: str = "Feature Importance"):
    """Create an interactive bar chart for feature importance"""
    df = pd.DataFrame({
        'Feature': feature_names[:top_n],
        'Importance': importances[:top_n]
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(color=df['Importance'], colorscale='Blues', showscale=True)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(df) * 25),
        width=800
    )
    return fig


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """Plot class distribution"""
    counts = y.value_counts().sort_index()
    labels = ["Low", "Medium", "High"]
    colors = ["#4caf50", "#ff9800", "#f44336"]
    
    fig = go.Figure(data=go.Bar(
        x=[labels[int(i)] if i < len(labels) else f"Class {i}" for i in counts.index],
        y=counts.values,
        marker=dict(color=[colors[int(i)] if i < len(colors) else "#2196f3" for i in counts.index])
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Delay Risk Class",
        yaxis_title="Count",
        width=600,
        height=400
    )
    return fig


def plot_probability_distribution(probs: Dict[str, float], title: str = "Prediction Probabilities"):
    """Plot probability distribution for a prediction"""
    labels = list(probs.keys())
    values = list(probs.values())
    colors = ["#4caf50", "#ff9800", "#f44336"]
    
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors[:len(values)])
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        width=600,
        height=400
    )
    return fig


def plot_time_series(df: pd.DataFrame, time_col: str, value_col: str, title: str = "Time Series"):
    """Plot time series data"""
    if time_col not in df.columns or value_col not in df.columns:
        return None
    
    df_plot = df[[time_col, value_col]].copy()
    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors='coerce')
    df_plot = df_plot.dropna()
    df_plot = df_plot.sort_values(time_col)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot[time_col],
        y=df_plot[value_col],
        mode='lines+markers',
        name=value_col,
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=value_col,
        width=1000,
        height=400,
        hovermode='x unified'
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation", max_features: int = 30):
    """Plot correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > max_features:
        # Select top features by variance
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_features).index.tolist()
    
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        text=corr.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 8},
        showscale=True
    ))
    fig.update_layout(
        title=title,
        width=1000,
        height=1000,
        xaxis=dict(tickangle=-45)
    )
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], title: str = "Model Comparison"):
    """Plot metrics comparison across models"""
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys()) if models else []
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        shared_yaxis=False
    )
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[m].get(metric, 0) for m in models]
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric, marker_color='#1f77b4'),
            row=1, col=i+1
        )
        fig.update_xaxes(title_text="Model", row=1, col=i+1)
        fig.update_yaxes(title_text=metric, row=1, col=i+1)
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False
    )
    return fig


def load_model_metrics() -> Dict:
    """Load saved model metrics if available"""
    metrics_path = os.path.join("output", "models", "lstm_delay_risk_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def format_number(num: float, decimals: int = 2) -> str:
    """Format number for display"""
    return f"{num:.{decimals}f}"


def get_risk_color(risk_level: int) -> str:
    """Get color for risk level"""
    colors = {0: "#4caf50", 1: "#ff9800", 2: "#f44336"}
    return colors.get(risk_level, "#757575")


def get_risk_label(risk_level: int) -> str:
    """Get label for risk level"""
    labels = {0: "Low", 1: "Medium", 2: "High"}
    return labels.get(risk_level, "Unknown")

