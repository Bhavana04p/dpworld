"""
FastAPI Backend for Port Congestion & Vessel Turnaround Optimization System

Provides REST APIs for:
- Delay risk prediction
- Resource optimization recommendations
- Health checks

All endpoints integrate with PostgreSQL database.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.schemas import (
    HealthResponse,
    DelayRiskPredictionRequest,
    DelayRiskPredictionResponse,
    OptimizationRequest,
    OptimizationResponse,
    ErrorResponse
)
from api.services.prediction_service import predict_delay_risk
from api.services.optimization_service import optimize_resources
from api.database import test_connection

# Initialize FastAPI app
app = FastAPI(
    title="Port Congestion & Vessel Turnaround Optimization API",
    description="REST API for delay risk prediction and resource optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Port Congestion & Vessel Turnaround Optimization API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Status of the API and database connection
    """
    try:
        db_connected = test_connection()
        return {
            "status": "ok",
            "database": "connected" if db_connected else "disconnected"
        }
    except Exception as e:
        return {
            "status": "ok",
            "database": f"error: {str(e)[:50]}"
        }


@app.post(
    "/predict-delay-risk",
    response_model=DelayRiskPredictionResponse,
    tags=["Prediction"],
    summary="Predict delay risk",
    description="Predict 24-hour delay risk based on port operational features"
)
async def predict_delay_risk_endpoint(
    request: DelayRiskPredictionRequest,
    save_to_db: bool = False
):
    """
    Predict delay risk for given operational conditions
    
    Args:
        request: Prediction request with feature values
        save_to_db: Whether to save prediction to database (default: False)
        
    Returns:
        Delay risk prediction with confidence and probabilities
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert request to dictionary
        request_dict = request.dict(exclude_none=True)
        
        # Make prediction
        result = predict_delay_risk(request_dict, save_to_db=save_to_db)
        
        return DelayRiskPredictionResponse(**result)
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/optimize-resources",
    response_model=OptimizationResponse,
    tags=["Optimization"],
    summary="Optimize resource allocation",
    description="Get optimization recommendations for crane allocation and yard utilization"
)
async def optimize_resources_endpoint(
    request: OptimizationRequest,
    save_to_db: bool = False
):
    """
    Optimize resource allocation to minimize delay risk
    
    Args:
        request: Optimization request with current conditions
        save_to_db: Whether to save optimization results to database (default: False)
        
    Returns:
        Optimization recommendations with before/after metrics
        
    Raises:
        HTTPException: If optimization fails
    """
    try:
        # Convert request to dictionary
        request_dict = request.dict(exclude_none=True)
        
        # Run optimization
        result = optimize_resources(request_dict, save_to_db=save_to_db)
        
        return OptimizationResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Required data not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/models/info", tags=["Models"])
async def get_model_info():
    """
    Get information about loaded ML models
    
    Returns:
        Model information including type and features
    """
    try:
        from api.services.prediction_service import load_model
        
        model, feature_names = load_model()
        
        return {
            "model_type": type(model).__name__,
            "model_available": True,
            "num_features": len(feature_names),
            "features": feature_names[:10]  # First 10 features
        }
    except Exception as e:
        return {
            "model_type": "Unknown",
            "model_available": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

