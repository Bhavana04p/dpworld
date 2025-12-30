"""
Historical Data Backfill Script for Power BI
Generates past 30 days of predictions and optimization data

This script creates historical data to populate Power BI dashboards with
sufficient data volume for meaningful visualizations.

Run once: python scripts/backfill_historical_data.py
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.db_connection import get_db_session, test_connection
from database.db_operations import (
    save_prediction,
    save_optimization_run,
    save_optimization_recommendations
)

# Try to load ML model
MODEL_AVAILABLE = False
PREDICT_FUNCTION = None

try:
    from api.services.prediction_service import predict_delay_risk, load_model
    try:
        model, features = load_model()
        MODEL_AVAILABLE = True
        PREDICT_FUNCTION = predict_delay_risk
        print(f"[INFO] ML model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"[INFO] ML model not available, using simulated predictions: {e}")
except Exception:
    print("[INFO] Using simulated predictions")


def generate_realistic_port_features_for_date(target_date: datetime) -> dict:
    """
    Generate realistic port operational features for a specific date
    
    Args:
        target_date: Target datetime for the features
        
    Returns:
        Dictionary with feature values
    """
    hour = target_date.hour
    day_of_week = target_date.weekday()
    is_weekend = day_of_week >= 5
    
    # Yard utilization varies by time of day and day of week
    if is_weekend:
        yard_util_base = random.uniform(0.35, 0.65)
    elif 8 <= hour <= 18:  # Business hours
        yard_util_base = random.uniform(0.65, 0.95)
    else:  # Off hours
        yard_util_base = random.uniform(0.40, 0.75)
    
    # Add some realistic variation
    yard_util_base += random.uniform(-0.05, 0.05)
    yard_util_base = max(0.2, min(0.98, yard_util_base))
    
    # Truck wait time correlates with yard utilization
    truck_wait = yard_util_base * 60 + random.uniform(-15, 25)
    truck_wait = max(10, min(150, truck_wait))
    
    # Weather varies by season (simplified)
    month = target_date.month
    if month in [12, 1, 2]:  # Winter
        wind_speed = random.uniform(10, 30)
        rainfall = random.uniform(0, 8)
    elif month in [6, 7, 8]:  # Summer
        wind_speed = random.uniform(5, 20)
        rainfall = random.uniform(0, 3)
    else:  # Spring/Fall
        wind_speed = random.uniform(8, 25)
        rainfall = random.uniform(0, 5)
    
    return {
        'yard_utilization_ratio': yard_util_base,
        'avg_truck_wait_min': truck_wait,
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'wave_height': random.uniform(0.5, 3.5),
        'temperature': random.uniform(10, 40),
        'visibility': random.uniform(5, 25),
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'month': month,
        'available_cranes': random.randint(10, 20),
        'berth_utilization': random.uniform(0.4, 0.9)
    }


def generate_prediction_for_date(features: dict, target_date: datetime) -> dict:
    """
    Generate a prediction for a specific date
    
    Args:
        features: Feature dictionary
        target_date: Target datetime
        
    Returns:
        Prediction result dictionary
    """
    if MODEL_AVAILABLE and PREDICT_FUNCTION is not None:
        try:
            # Use real ML model
            result = PREDICT_FUNCTION(features, save_to_db=False)
            return result
        except Exception:
            pass
    
    # Fallback: Simulate prediction based on features
    yard_util = features.get('yard_utilization_ratio', 0.5)
    truck_wait = features.get('avg_truck_wait_min', 30)
    
    # Simple heuristic: higher utilization + longer wait = higher risk
    risk_score = (yard_util * 0.6) + (min(truck_wait / 100, 1.0) * 0.4)
    
    if risk_score < 0.4:
        delay_risk = 0  # Low
        probabilities = {"low": 0.7, "medium": 0.25, "high": 0.05}
    elif risk_score < 0.7:
        delay_risk = 1  # Medium
        probabilities = {"low": 0.25, "medium": 0.6, "high": 0.15}
    else:
        delay_risk = 2  # High
        probabilities = {"low": 0.1, "medium": 0.3, "high": 0.6}
    
    confidence = max(probabilities.values())
    
    return {
        "delay_risk": delay_risk,
        "confidence": confidence,
        "probabilities": probabilities
    }


def generate_optimization_for_date(features: dict, current_risk: float, target_date: datetime) -> dict:
    """
    Generate optimization recommendations for a specific date
    
    Args:
        features: Feature dictionary
        current_risk: Current delay risk value
        target_date: Target datetime
        
    Returns:
        Optimization result dictionary
    """
    current_yard_util = features.get('yard_utilization_ratio', 0.85)
    available_cranes = features.get('available_cranes', 20)
    
    # Simple optimization logic
    if current_yard_util > 0.85:
        recommended_yard_util = max(0.70, current_yard_util - 0.15)
        recommended_cranes = min(available_cranes, int(available_cranes * 1.2))
    elif current_yard_util < 0.60:
        recommended_yard_util = current_yard_util + 0.10
        recommended_cranes = max(2, int(available_cranes * 0.9))
    else:
        recommended_yard_util = current_yard_util
        recommended_cranes = available_cranes
    
    delay_risk_after = max(0.0, current_risk * 0.6)
    improvement = ((current_risk - delay_risk_after) / current_risk * 100) if current_risk > 0 else 0
    
    return {
        'status': 'optimal',
        'solver_used': 'Historical',
        'recommended_cranes': recommended_cranes,
        'yard_utilization_target': recommended_yard_util,
        'delay_risk_before': current_risk,
        'delay_risk_after': delay_risk_after,
        'improvement_percent': improvement,
        'recommendations': [{
            'window_id': 0,
            'start_time': target_date,
            'end_time': target_date + timedelta(hours=6),
            'current_yard_util': current_yard_util,
            'recommended_yard_util': recommended_yard_util,
            'recommended_cranes': recommended_cranes,
            'current_delay_risk': current_risk,
            'expected_risk_reduction': current_risk - delay_risk_after
        }]
    }


def backfill_historical_data(days_back: int = 30, interval_minutes: int = 15):
    """
    Backfill historical data for the specified number of days
    
    Args:
        days_back: Number of days to go back (default: 30)
        interval_minutes: Interval between data points in minutes (default: 15)
    """
    print("=" * 70)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 70)
    
    # Test database connection
    print("\n[1/4] Testing database connection...")
    if not test_connection():
        print("[ERROR] Cannot connect to database!")
        return False
    print("[SUCCESS] Database connected!")
    
    # Calculate data points
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    # Generate timestamps every interval_minutes
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)
    
    total_points = len(timestamps)
    print(f"\n[2/4] Data generation plan:")
    print(f"   Start date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Interval: {interval_minutes} minutes")
    print(f"   Total data points: {total_points:,}")
    print(f"   Estimated rows: ~{total_points:,} predictions")
    
    # Confirm
    print("\n[3/4] Starting backfill...")
    print("   This may take 1-3 minutes. Please wait...")
    
    predictions_saved = 0
    optimizations_saved = 0
    
    try:
        with get_db_session() as session:
            for i, target_date in enumerate(timestamps, 1):
                # Generate features for this timestamp
                features = generate_realistic_port_features_for_date(target_date)
                
                # Generate prediction
                prediction = generate_prediction_for_date(features, target_date)
                
                # Save prediction
                try:
                    prediction_obj = save_prediction(
                        session=session,
                        timestamp=target_date,
                        time_window_start=target_date,
                        predicted_risk_class=prediction['delay_risk'],
                        probability_low=prediction['probabilities']['low'],
                        probability_medium=prediction['probabilities']['medium'],
                        probability_high=prediction['probabilities']['high'],
                        model_type="RandomForest" if MODEL_AVAILABLE else "Simulated",
                        yard_utilization_ratio=features['yard_utilization_ratio'],
                        avg_truck_wait_min=features['avg_truck_wait_min']
                    )
                    predictions_saved += 1
                except Exception as e:
                    print(f"[WARNING] Failed to save prediction for {target_date}: {e}")
                    continue
                
                # Generate optimization every 4th data point (every hour)
                if i % 4 == 0:
                    current_risk = float(prediction['delay_risk']) + prediction['probabilities']['high']
                    optimization = generate_optimization_for_date(features, current_risk, target_date)
                    
                    try:
                        run_id = target_date.strftime("%Y%m%d_%H%M%S")
                        
                        opt_run = save_optimization_run(
                            session=session,
                            run_id=run_id,
                            status=optimization['status'],
                            solver_used=optimization['solver_used'],
                            config={
                                'max_available_cranes': features.get('available_cranes', 20),
                                'min_cranes_per_window': 2,
                                'safe_yard_utilization': 0.80,
                                'max_yard_utilization': 0.95
                            },
                            total_windows=1,
                            total_recommendations=len(optimization['recommendations']),
                            objective_value=None,
                            impact_metrics={
                                'before': {'avg_delay_risk': optimization['delay_risk_before']},
                                'after': {'avg_delay_risk': optimization['delay_risk_after']},
                                'improvements': {
                                    'delay_risk_reduction_pct': optimization['improvement_percent']
                                }
                            }
                        )
                        
                        save_optimization_recommendations(
                            session=session,
                            recommendations=optimization['recommendations'],
                            optimization_run_id=run_id,
                            optimization_status=optimization['status'],
                            objective_value=None
                        )
                        optimizations_saved += 1
                    except Exception as e:
                        print(f"[WARNING] Failed to save optimization for {target_date}: {e}")
                
                # Progress indicator
                if i % 100 == 0:
                    progress = (i / total_points) * 100
                    print(f"   Progress: {i:,}/{total_points:,} ({progress:.1f}%) - Saved {predictions_saved:,} predictions, {optimizations_saved:,} optimizations")
            
            # Commit all changes
            session.commit()
            print(f"\n   Final commit: {predictions_saved:,} predictions, {optimizations_saved:,} optimizations")
    
    except Exception as e:
        print(f"\n[ERROR] Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[4/4] Backfill complete!")
    print("=" * 70)
    print(f"[SUCCESS] Historical data backfill completed!")
    print(f"   Predictions saved: {predictions_saved:,}")
    print(f"   Optimizations saved: {optimizations_saved:,}")
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\n[INFO] Next steps:")
    print("   1. Verify data in PostgreSQL:")
    print("      SELECT COUNT(*) FROM predictions;")
    print("   2. Refresh Power BI dataset")
    print("   3. Create visualizations with historical data")
    print("=" * 70)
    
    return True


def main():
    """Main function"""
    # Configuration
    DAYS_BACK = 30  # 30 days of historical data
    INTERVAL_MINUTES = 15  # Data point every 15 minutes
    
    # Calculate expected rows
    expected_rows = (DAYS_BACK * 24 * 60) // INTERVAL_MINUTES
    print(f"\nExpected rows: ~{expected_rows:,} predictions")
    print(f"Expected optimization runs: ~{expected_rows // 4:,}")
    
    success = backfill_historical_data(days_back=DAYS_BACK, interval_minutes=INTERVAL_MINUTES)
    
    if success:
        print("\n[SUCCESS] Historical backfill completed successfully!")
        print("\nVerify data:")
        print("  psql -U postgres -d port_congestion_db")
        print("  SELECT COUNT(*) FROM predictions;")
        print("  SELECT COUNT(*) FROM optimization_runs;")
    else:
        print("\n[ERROR] Historical backfill failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

