"""
Database operations for saving and retrieving predictions and recommendations
"""
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
import pandas as pd

from database.models import (
    Prediction, OptimizationRecommendation, OperationalDecision, OptimizationRun
)
from database.db_connection import get_db_session


def save_prediction(
    session: Session,
    timestamp: datetime,
    time_window_start: datetime,
    predicted_risk_class: int,
    probability_low: float,
    probability_medium: float,
    probability_high: float,
    model_type: str = "RandomForest",
    model_version: Optional[str] = None,
    yard_utilization_ratio: Optional[float] = None,
    avg_truck_wait_min: Optional[float] = None,
    crane_utilization_ratio: Optional[float] = None,
    time_window_end: Optional[datetime] = None,
    notes: Optional[str] = None
) -> Prediction:
    """
    Save a prediction to the database
    
    Returns:
        Created Prediction object
    """
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    
    prediction = Prediction(
        timestamp=timestamp,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        predicted_risk_class=predicted_risk_class,
        predicted_risk_label=risk_labels.get(predicted_risk_class, "Unknown"),
        probability_low=probability_low,
        probability_medium=probability_medium,
        probability_high=probability_high,
        max_probability=max(probability_low, probability_medium, probability_high),
        yard_utilization_ratio=yard_utilization_ratio,
        avg_truck_wait_min=avg_truck_wait_min,
        crane_utilization_ratio=crane_utilization_ratio,
        model_type=model_type,
        model_version=model_version,
        notes=notes
    )
    
    session.add(prediction)
    session.flush()  # Get the ID
    return prediction


def save_optimization_run(
    session: Session,
    run_id: str,
    status: str,
    solver_used: Optional[str],
    config: Dict,
    total_windows: int,
    total_recommendations: int,
    objective_value: Optional[float],
    impact_metrics: Dict,
    notes: Optional[str] = None
) -> OptimizationRun:
    """
    Save optimization run metadata
    
    Returns:
        Created OptimizationRun object
    """
    run = OptimizationRun(
        run_id=run_id,
        run_timestamp=datetime.utcnow(),
        status=status,
        solver_used=solver_used,
        max_available_cranes=config.get('max_available_cranes', 20),
        min_cranes_per_window=config.get('min_cranes_per_window', 2),
        safe_yard_utilization=config.get('safe_yard_utilization', 0.80),
        max_yard_utilization=config.get('max_yard_utilization', 0.95),
        risk_weight=config.get('risk_weight', 10.0),
        congestion_weight=config.get('congestion_weight', 5.0),
        resource_cost_weight=config.get('resource_cost_weight', 1.0),
        total_windows=total_windows,
        total_recommendations=total_recommendations,
        objective_value=objective_value,
        before_avg_delay_risk=impact_metrics.get('before', {}).get('avg_delay_risk'),
        after_avg_delay_risk=impact_metrics.get('after', {}).get('avg_delay_risk'),
        delay_risk_reduction_pct=impact_metrics.get('improvements', {}).get('delay_risk_reduction_pct'),
        high_risk_windows_reduced=impact_metrics.get('improvements', {}).get('high_risk_reduction'),
        yard_util_improvement_pct=impact_metrics.get('improvements', {}).get('yard_util_improvement_pct'),
        notes=notes
    )
    
    session.add(run)
    session.flush()
    return run


def save_optimization_recommendations(
    session: Session,
    recommendations: List[Dict],
    optimization_run_id: str,
    optimization_status: str,
    objective_value: Optional[float] = None,
    prediction_id: Optional[int] = None
) -> List[OptimizationRecommendation]:
    """
    Save optimization recommendations to database
    
    Args:
        session: Database session
        recommendations: List of recommendation dictionaries
        optimization_run_id: ID of the optimization run
        optimization_status: Status of optimization
        objective_value: Objective function value
        prediction_id: Optional prediction ID to link
    
    Returns:
        List of created OptimizationRecommendation objects
    """
    saved_recommendations = []
    
    for rec in recommendations:
        # Parse datetime strings safely
        start_time_str = str(rec.get('start_time', datetime.utcnow()))
        end_time_str = str(rec.get('end_time', datetime.utcnow()))
        
        # Handle various datetime formats
        try:
            if isinstance(rec.get('start_time'), datetime):
                start_time = rec.get('start_time')
            else:
                start_time_str = start_time_str.replace('Z', '+00:00').replace('+00:00', '')
                start_time = pd.to_datetime(start_time_str).to_pydatetime()
        except Exception:
            start_time = datetime.utcnow()
        
        try:
            if isinstance(rec.get('end_time'), datetime):
                end_time = rec.get('end_time')
            else:
                end_time_str = end_time_str.replace('Z', '+00:00').replace('+00:00', '')
                end_time = pd.to_datetime(end_time_str).to_pydatetime()
        except Exception:
            end_time = datetime.utcnow()
        
        recommendation = OptimizationRecommendation(
            prediction_id=prediction_id,
            window_id=int(rec.get('window_id', 0)),
            start_time=start_time,
            end_time=end_time,
            current_yard_util=float(rec.get('current_yard_util', 0)),
            current_delay_risk=float(rec.get('current_delay_risk', 0)),
            current_truck_wait=rec.get('current_truck_wait'),
            recommended_yard_util=float(rec.get('recommended_yard_util', 0)),
            recommended_cranes=int(rec.get('recommended_cranes', 0)),
            expected_risk_reduction=float(rec.get('expected_risk_reduction', 0)),
            optimization_run_id=optimization_run_id,
            optimization_status=optimization_status,
            objective_value=objective_value
        )
        
        session.add(recommendation)
        saved_recommendations.append(recommendation)
    
    session.flush()
    return saved_recommendations


def save_operational_decision(
    session: Session,
    recommendation_id: Optional[int],
    decision_type: str,
    actual_cranes_allocated: Optional[int] = None,
    actual_yard_util_target: Optional[float] = None,
    actual_actions: Optional[str] = None,
    decision_maker: Optional[str] = None,
    approval_status: Optional[str] = None
) -> OperationalDecision:
    """
    Save an operational decision
    
    Returns:
        Created OperationalDecision object
    """
    decision = OperationalDecision(
        recommendation_id=recommendation_id,
        decision_timestamp=datetime.utcnow(),
        decision_type=decision_type,
        actual_cranes_allocated=actual_cranes_allocated,
        actual_yard_util_target=actual_yard_util_target,
        actual_actions=actual_actions,
        decision_maker=decision_maker,
        approval_status=approval_status
    )
    
    session.add(decision)
    session.flush()
    return decision


def get_latest_predictions(session: Session, limit: int = 100) -> List[Prediction]:
    """Get latest predictions"""
    return session.query(Prediction).order_by(desc(Prediction.timestamp)).limit(limit).all()


def get_predictions_by_date_range(
    session: Session,
    start_date: datetime,
    end_date: datetime
) -> List[Prediction]:
    """Get predictions within date range"""
    return session.query(Prediction).filter(
        and_(
            Prediction.timestamp >= start_date,
            Prediction.timestamp <= end_date
        )
    ).order_by(desc(Prediction.timestamp)).all()


def get_latest_optimization_run(session: Session) -> Optional[OptimizationRun]:
    """Get the most recent optimization run"""
    return session.query(OptimizationRun).order_by(desc(OptimizationRun.run_timestamp)).first()


def get_optimization_recommendations_by_run(
    session: Session,
    run_id: str
) -> List[OptimizationRecommendation]:
    """Get all recommendations for a specific optimization run"""
    return session.query(OptimizationRecommendation).filter(
        OptimizationRecommendation.optimization_run_id == run_id
    ).order_by(OptimizationRecommendation.start_time).all()


def get_unimplemented_recommendations(session: Session) -> List[OptimizationRecommendation]:
    """Get recommendations that haven't been implemented yet"""
    return session.query(OptimizationRecommendation).filter(
        OptimizationRecommendation.implemented == False
    ).order_by(OptimizationRecommendation.start_time).all()

