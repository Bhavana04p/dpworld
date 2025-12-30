"""
SQLAlchemy Models for Port Congestion Database
Stores predictions, optimization recommendations, and timestamped decisions
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import Optional

Base = declarative_base()


class Prediction(Base):
    """Store ML model predictions for delay risk"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    time_window_start = Column(DateTime, nullable=False, index=True)
    time_window_end = Column(DateTime, nullable=True)
    
    # Prediction results
    predicted_risk_class = Column(Integer, nullable=False)  # 0=Low, 1=Medium, 2=High
    predicted_risk_label = Column(String(20), nullable=False)  # 'Low', 'Medium', 'High'
    probability_low = Column(Float, nullable=False)
    probability_medium = Column(Float, nullable=False)
    probability_high = Column(Float, nullable=False)
    max_probability = Column(Float, nullable=False)
    
    # Input features (key ones)
    yard_utilization_ratio = Column(Float, nullable=True)
    avg_truck_wait_min = Column(Float, nullable=True)
    crane_utilization_ratio = Column(Float, nullable=True)
    
    # Model info
    model_type = Column(String(50), nullable=False)  # 'RandomForest', 'XGBoost', 'Heuristic', etc.
    model_version = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    
    # Relationships
    recommendations = relationship("OptimizationRecommendation", back_populates="prediction")
    
    __table_args__ = (
        Index('idx_pred_timestamp', 'timestamp'),
        Index('idx_pred_risk_class', 'predicted_risk_class'),
    )


class OptimizationRecommendation(Base):
    """Store optimization recommendations for resource allocation"""
    __tablename__ = 'optimization_recommendations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=True)
    
    # Time window
    window_id = Column(Integer, nullable=False)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=False)
    
    # Current state
    current_yard_util = Column(Float, nullable=False)
    current_delay_risk = Column(Float, nullable=False)
    current_truck_wait = Column(Float, nullable=True)
    
    # Recommendations
    recommended_yard_util = Column(Float, nullable=False)
    recommended_cranes = Column(Integer, nullable=False)
    expected_risk_reduction = Column(Float, nullable=False)
    
    # Optimization run info
    optimization_run_id = Column(String(100), nullable=False, index=True)  # Timestamp-based ID
    optimization_status = Column(String(20), nullable=False)  # 'optimal', 'heuristic', 'infeasible'
    objective_value = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    implemented = Column(Boolean, default=False, nullable=False)
    implementation_notes = Column(Text, nullable=True)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="recommendations")
    decisions = relationship("OperationalDecision", back_populates="recommendation")
    
    __table_args__ = (
        Index('idx_opt_run_id', 'optimization_run_id'),
        Index('idx_opt_start_time', 'start_time'),
        Index('idx_opt_implemented', 'implemented'),
    )


class OperationalDecision(Base):
    """Store actual operational decisions made based on recommendations"""
    __tablename__ = 'operational_decisions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(Integer, ForeignKey('optimization_recommendations.id'), nullable=True)
    
    # Decision details
    decision_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    decision_type = Column(String(50), nullable=False)  # 'crane_allocation', 'yard_utilization', 'resource_reallocation'
    
    # Actual actions taken
    actual_cranes_allocated = Column(Integer, nullable=True)
    actual_yard_util_target = Column(Float, nullable=True)
    actual_actions = Column(Text, nullable=True)  # JSON or text description
    
    # Decision maker
    decision_maker = Column(String(100), nullable=True)  # User/system identifier
    approval_status = Column(String(20), nullable=True)  # 'approved', 'pending', 'rejected'
    
    # Outcomes (filled after implementation)
    actual_delay_risk = Column(Float, nullable=True)
    actual_yard_util = Column(Float, nullable=True)
    outcome_notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    recommendation = relationship("OptimizationRecommendation", back_populates="decisions")
    
    __table_args__ = (
        Index('idx_decision_timestamp', 'decision_timestamp'),
        Index('idx_decision_type', 'decision_type'),
    )


class OptimizationRun(Base):
    """Store metadata about each optimization run"""
    __tablename__ = 'optimization_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)  # Timestamp-based ID
    
    # Run metadata
    run_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    status = Column(String(20), nullable=False)  # 'optimal', 'heuristic', 'infeasible'
    solver_used = Column(String(50), nullable=True)  # 'OR-Tools', 'PuLP', 'Heuristic'
    
    # Configuration
    max_available_cranes = Column(Integer, nullable=False)
    min_cranes_per_window = Column(Integer, nullable=False)
    safe_yard_utilization = Column(Float, nullable=False)
    max_yard_utilization = Column(Float, nullable=False)
    risk_weight = Column(Float, nullable=False)
    congestion_weight = Column(Float, nullable=False)
    resource_cost_weight = Column(Float, nullable=False)
    
    # Results summary
    total_windows = Column(Integer, nullable=False)
    total_recommendations = Column(Integer, nullable=False)
    objective_value = Column(Float, nullable=True)
    
    # Impact metrics
    before_avg_delay_risk = Column(Float, nullable=True)
    after_avg_delay_risk = Column(Float, nullable=True)
    delay_risk_reduction_pct = Column(Float, nullable=True)
    high_risk_windows_reduced = Column(Integer, nullable=True)
    yard_util_improvement_pct = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_run_timestamp', 'run_timestamp'),
        Index('idx_run_status', 'status'),
    )

