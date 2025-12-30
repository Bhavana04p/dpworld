"""
Pydantic schemas for request/response validation
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"


class DelayRiskPredictionRequest(BaseModel):
    """Request schema for delay risk prediction"""
    yard_utilization_ratio: float = Field(..., ge=0.0, le=1.0, description="Yard utilization ratio (0-1)")
    avg_truck_wait_min: float = Field(..., ge=0.0, description="Average truck wait time in minutes")
    
    # Optional weather features
    wind_speed: Optional[float] = Field(None, ge=0.0, description="Wind speed")
    rainfall: Optional[float] = Field(None, ge=0.0, description="Rainfall amount")
    wave_height: Optional[float] = Field(None, ge=0.0, description="Wave height")
    temperature: Optional[float] = Field(None, description="Temperature")
    visibility: Optional[float] = Field(None, ge=0.0, description="Visibility")
    
    # Optional temporal features
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: Optional[bool] = Field(None, description="Is weekend")
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    
    # Optional crane/berth features
    available_cranes: Optional[int] = Field(None, ge=0, description="Number of available cranes")
    berth_utilization: Optional[float] = Field(None, ge=0.0, le=1.0, description="Berth utilization ratio")
    
    class Config:
        schema_extra = {
            "example": {
                "yard_utilization_ratio": 0.85,
                "avg_truck_wait_min": 45.5,
                "wind_speed": 15.0,
                "hour_of_day": 14,
                "day_of_week": 2,
                "is_weekend": False
            }
        }


class DelayRiskPredictionResponse(BaseModel):
    """Response schema for delay risk prediction"""
    delay_risk: int = Field(..., ge=0, le=2, description="Delay risk class: 0=Low, 1=Medium, 2=High")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    prediction_id: Optional[int] = Field(None, description="Database ID if saved")
    
    class Config:
        schema_extra = {
            "example": {
                "delay_risk": 1,
                "confidence": 0.75,
                "probabilities": {
                    "low": 0.15,
                    "medium": 0.75,
                    "high": 0.10
                },
                "prediction_id": 123
            }
        }


class OptimizationRequest(BaseModel):
    """Request schema for resource optimization"""
    time_window_hours: int = Field(24, ge=1, le=168, description="Time window in hours (default 24)")
    current_yard_utilization: float = Field(..., ge=0.0, le=1.0, description="Current yard utilization")
    available_cranes: int = Field(..., ge=1, description="Number of available cranes")
    current_delay_risk: Optional[float] = Field(None, ge=0.0, description="Current delay risk (optional)")
    avg_truck_wait_min: Optional[float] = Field(None, ge=0.0, description="Average truck wait time (optional)")
    
    # Configuration overrides (optional)
    max_available_cranes: Optional[int] = Field(None, ge=1, description="Max available cranes (overrides available_cranes)")
    min_cranes_per_window: Optional[int] = Field(None, ge=1, description="Minimum cranes per window")
    safe_yard_utilization: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safe yard utilization threshold")
    max_yard_utilization: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum yard utilization")
    
    class Config:
        schema_extra = {
            "example": {
                "time_window_hours": 24,
                "current_yard_utilization": 0.85,
                "available_cranes": 20,
                "current_delay_risk": 1.5,
                "avg_truck_wait_min": 45.0
            }
        }


class OptimizationRecommendation(BaseModel):
    """Single optimization recommendation"""
    window_id: int
    start_time: datetime
    end_time: datetime
    current_yard_util: float
    recommended_yard_util: float
    recommended_cranes: int
    current_delay_risk: float
    expected_risk_reduction: float


class OptimizationResponse(BaseModel):
    """Response schema for resource optimization"""
    status: str = Field(..., description="Optimization status: optimal, heuristic, or infeasible")
    solver_used: str = Field(..., description="Solver used: OR-Tools, PuLP, or Heuristic")
    recommended_cranes: int = Field(..., ge=0, description="Recommended number of cranes")
    yard_utilization_target: float = Field(..., ge=0.0, le=1.0, description="Target yard utilization")
    delay_risk_before: float = Field(..., ge=0.0, description="Delay risk before optimization")
    delay_risk_after: float = Field(..., ge=0.0, description="Delay risk after optimization")
    improvement_percent: float = Field(..., description="Percentage improvement in delay risk")
    recommendations: List[OptimizationRecommendation] = Field(..., description="Detailed recommendations per window")
    optimization_run_id: Optional[str] = Field(None, description="Database run ID if saved")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "optimal",
                "solver_used": "OR-Tools",
                "recommended_cranes": 15,
                "yard_utilization_target": 0.75,
                "delay_risk_before": 1.5,
                "delay_risk_after": 0.8,
                "improvement_percent": 46.7,
                "recommendations": [],
                "optimization_run_id": "20251229_001050"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

