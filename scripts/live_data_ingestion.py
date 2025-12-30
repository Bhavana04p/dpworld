"""
Live Data Ingestion Script for Power BI Integration
Periodically writes predictions and optimization results to PostgreSQL

This script simulates live port operations by:
1. Generating predictions using the trained ML model
2. Running optimization recommendations
3. Saving everything to PostgreSQL tables
4. Power BI can then refresh and display this data

Run continuously: python scripts/live_data_ingestion.py
"""
import sys
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict

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

# Try to load ML model for real predictions
MODEL_AVAILABLE = False
MODEL_LOADED = None
MODEL_FEATURES = []
PREDICT_FUNCTION = None

try:
    from api.services.prediction_service import predict_delay_risk, load_model
    PREDICT_FUNCTION = predict_delay_risk
    try:
        MODEL_LOADED, MODEL_FEATURES = load_model()
        MODEL_AVAILABLE = True
    except Exception:
        # Model not found - will use simulation
        MODEL_AVAILABLE = False
except Exception:
    # Prediction service not available - will use simulation
    MODEL_AVAILABLE = False

# Try to load optimization logic
try:
    from scripts.optimize_resources import (
        prepare_optimization_data,
        optimize_with_ortools,
        optimize_with_pulp,
        heuristic_optimization,
        calculate_impact_analysis
    )
    OPTIMIZATION_AVAILABLE = True
except Exception:
    OPTIMIZATION_AVAILABLE = False
    print("[WARNING] Optimization module not available. Using simulated optimization.")

# Weather API Configuration
WEATHER_API_KEY = "707a25995ab04781ac761618252812"
# Default port location (can be configured)
DEFAULT_LATITUDE = 25.2048  # Dubai Port coordinates (adjust as needed)
DEFAULT_LONGITUDE = 55.2708

# Try multiple weather API endpoints
WEATHER_API_URLS = [
    "http://api.openweathermap.org/data/2.5/weather",  # OpenWeatherMap
    "https://api.openweathermap.org/data/2.5/weather",  # OpenWeatherMap HTTPS
    "http://api.weatherapi.com/v1/current.json",  # WeatherAPI.com alternative
]


def load_weather_from_csv(target_timestamp: Optional[datetime] = None) -> Optional[Dict]:
    """Load weather data from CSV file"""
    try:
        weather_file = project_root / "weather_data.csv"
        if not weather_file.exists():
            return None
        
        weather_df = pd.read_csv(weather_file, nrows=10000)  # Load sample
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')
        
        if target_timestamp is None:
            target_timestamp = datetime.now()
        
        # Find closest timestamp
        weather_df['time_diff'] = abs(weather_df['timestamp'] - target_timestamp)
        closest_idx = weather_df['time_diff'].idxmin()
        
        if pd.isna(closest_idx):
            return None
        
        row = weather_df.loc[closest_idx]
        
        return {
            'wind_speed': float(row.get('wind_speed_mps', 0)) * 3.6,  # m/s to km/h
            'wave_height': float(row.get('wave_height_m', 0)),
            'rainfall': float(row.get('rainfall_mm', 0)),
            'visibility': float(row.get('visibility_km', 10)),
            'temperature': float(row.get('temperature_c', 20))
        }
    except Exception:
        return None


def fetch_weather_data(lat: float = DEFAULT_LATITUDE, lon: float = DEFAULT_LONGITUDE) -> Optional[Dict]:
    """
    Fetch real-time weather data from weather API
    
    Tries multiple API endpoints and parameter formats to find working configuration
    
    Args:
        lat: Latitude (default: Dubai Port)
        lon: Longitude (default: Dubai Port)
        
    Returns:
        Dictionary with weather data or None if API fails
    """
    # Try different API parameter formats
    api_configs = [
        # OpenWeatherMap format
        {
            'url': 'https://api.openweathermap.org/data/2.5/weather',
            'params': {
                'lat': lat,
                'lon': lon,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
        },
        # Alternative: API key as query parameter
        {
            'url': f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric',
            'params': None
        },
        # WeatherAPI.com format (if different service)
        {
            'url': 'https://api.weatherapi.com/v1/current.json',
            'params': {
                'key': WEATHER_API_KEY,
                'q': f'{lat},{lon}'
            }
        }
    ]
    
    for config in api_configs:
        try:
            if config['params']:
                response = requests.get(config['url'], params=config['params'], timeout=5)
            else:
                response = requests.get(config['url'], timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Try to extract weather data (handle different API formats)
                weather_info = {}
                
                # OpenWeatherMap format
                if 'wind' in data and 'main' in data:
                    weather_info = {
                        'wind_speed': data.get('wind', {}).get('speed', 0) * 3.6,  # Convert m/s to km/h
                        'wind_direction': data.get('wind', {}).get('deg', 0),
                        'temperature': data.get('main', {}).get('temp', 20),
                        'humidity': data.get('main', {}).get('humidity', 50),
                        'pressure': data.get('main', {}).get('pressure', 1013),
                        'visibility': data.get('visibility', 10000) / 1000 if data.get('visibility') else 10,
                        'rainfall': data.get('rain', {}).get('1h', 0) if 'rain' in data else 0,
                        'clouds': data.get('clouds', {}).get('all', 0),
                        'weather_main': data.get('weather', [{}])[0].get('main', 'Clear'),
                        'weather_description': data.get('weather', [{}])[0].get('description', 'clear sky')
                    }
                # WeatherAPI.com format
                elif 'current' in data:
                    current = data['current']
                    weather_info = {
                        'wind_speed': current.get('wind_kph', 0),
                        'wind_direction': current.get('wind_degree', 0),
                        'temperature': current.get('temp_c', 20),
                        'humidity': current.get('humidity', 50),
                        'pressure': current.get('pressure_mb', 1013),
                        'visibility': current.get('vis_km', 10),
                        'rainfall': current.get('precip_mm', 0),
                        'clouds': current.get('cloud', 0),
                        'weather_main': current.get('condition', {}).get('text', 'Clear'),
                        'weather_description': current.get('condition', {}).get('text', 'clear sky')
                    }
                else:
                    continue  # Try next config
                
                # Estimate wave height based on wind speed
                wind_speed_ms = weather_info.get('wind_speed', 0) / 3.6  # Convert km/h to m/s
                weather_info['wave_height'] = max(0.5, min(4.0, wind_speed_ms * 0.3))
                
                return weather_info
            elif response.status_code == 401:
                # Unauthorized - try next config
                continue
            else:
                # Other error - try next config
                continue
                
        except requests.exceptions.RequestException:
            # Network error - try next config
            continue
        except Exception:
            # Parse error - try next config
            continue
    
    # All configs failed
    return None


def generate_realistic_port_features() -> dict:
    """
    Generate realistic port operational features
    Uses real weather API data when available, falls back to simulation
    
    Returns:
        Dictionary with feature values
    """
    # Simulate realistic port operations
    hour = datetime.now().hour
    
    # Yard utilization varies by time of day (higher during business hours)
    if 8 <= hour <= 18:
        yard_util_base = random.uniform(0.65, 0.95)
    else:
        yard_util_base = random.uniform(0.40, 0.75)
    
    # Truck wait time correlates with yard utilization
    truck_wait = yard_util_base * 60 + random.uniform(-10, 20)
    truck_wait = max(10, min(120, truck_wait))
    
    # Try to fetch weather from CSV first, then API
    weather_data = load_weather_from_csv(datetime.now())
    
    if not weather_data:
        # Try API
        weather_data = fetch_weather_data()
    
    # Use real weather if available, otherwise simulate
    if weather_data:
        wind_speed = weather_data.get('wind_speed', random.uniform(5, 25))
        rainfall = weather_data.get('rainfall', 0)
        wave_height = weather_data.get('wave_height', random.uniform(0.5, 3.0))
        temperature = weather_data.get('temperature', random.uniform(15, 35))
        visibility = weather_data.get('visibility', random.uniform(5, 20))
        weather_source = "API"
    else:
        # Fallback to simulated weather
        wind_speed = random.uniform(5, 25)
        rainfall = random.uniform(0, 5)
        wave_height = random.uniform(0.5, 3.0)
        temperature = random.uniform(15, 35)
        visibility = random.uniform(5, 20)
        weather_source = "Simulated"
    
    return {
        'yard_utilization_ratio': yard_util_base,
        'avg_truck_wait_min': truck_wait,
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'wave_height': wave_height,
        'temperature': temperature,
        'visibility': visibility,
        'hour_of_day': hour,
        'day_of_week': datetime.now().weekday(),
        'is_weekend': datetime.now().weekday() >= 5,
        'month': datetime.now().month,
        'available_cranes': random.randint(12, 20),
        'berth_utilization': random.uniform(0.5, 0.9),
        '_weather_source': weather_source  # For debugging
    }


def generate_prediction(features: dict) -> dict:
    """
    Generate a prediction using ML model or simulation
    
    Args:
        features: Feature dictionary
        
    Returns:
        Prediction result dictionary
    """
    if MODEL_AVAILABLE and PREDICT_FUNCTION is not None:
        try:
            # Use real ML model
            result = PREDICT_FUNCTION(features, save_to_db=False)
            return result
        except Exception as e:
            # Fallback to simulation if model fails
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


def generate_optimization_recommendations(features: dict, current_risk: float) -> dict:
    """
    Generate optimization recommendations
    
    Args:
        features: Feature dictionary
        current_risk: Current delay risk value
        
    Returns:
        Optimization result dictionary
    """
    if OPTIMIZATION_AVAILABLE:
        try:
            # Use real optimization logic
            request_data = {
                'time_window_hours': 24,
                'current_yard_utilization': features.get('yard_utilization_ratio', 0.85),
                'available_cranes': features.get('available_cranes', 20),
                'current_delay_risk': current_risk,
                'avg_truck_wait_min': features.get('avg_truck_wait_min', 45.0)
            }
            
            from api.services.optimization_service import optimize_resources
            result = optimize_resources(request_data, save_to_db=False)
            return result
        except Exception as e:
            print(f"[WARNING] Optimization failed: {e}. Using simulation.")
    
    # Fallback: Simulate optimization
    current_yard_util = features.get('yard_utilization_ratio', 0.85)
    available_cranes = features.get('available_cranes', 20)
    
    # Simple optimization logic
    if current_yard_util > 0.85:
        recommended_yard_util = max(0.70, current_yard_util - 0.15)
        recommended_cranes = min(available_cranes, int(available_cranes * 1.2))
    else:
        recommended_yard_util = current_yard_util
        recommended_cranes = available_cranes
    
    delay_risk_after = max(0.0, current_risk * 0.6)
    improvement = ((current_risk - delay_risk_after) / current_risk * 100) if current_risk > 0 else 0
    
    return {
        'status': 'optimal',
        'solver_used': 'Simulated',
        'recommended_cranes': recommended_cranes,
        'yard_utilization_target': recommended_yard_util,
        'delay_risk_before': current_risk,
        'delay_risk_after': delay_risk_after,
        'improvement_percent': improvement,
        'recommendations': [{
            'window_id': 0,
            'start_time': datetime.utcnow(),
            'end_time': datetime.utcnow() + timedelta(hours=6),
            'current_yard_util': current_yard_util,
            'recommended_yard_util': recommended_yard_util,
            'recommended_cranes': recommended_cranes,
            'current_delay_risk': current_risk,
            'expected_risk_reduction': current_risk - delay_risk_after
        }]
    }


def ingest_data_cycle():
    """
    Perform one cycle of data ingestion:
    1. Generate realistic port features
    2. Make prediction
    3. Generate optimization recommendations
    4. Save to PostgreSQL
    """
    try:
        # Test database connection
        if not test_connection():
            print("[ERROR] Database connection failed!")
            return False
        
        # Generate realistic features (with real weather API data)
        features = generate_realistic_port_features()
        weather_source = features.pop('_weather_source', 'Unknown')
        
        # Generate prediction
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating prediction...")
        prediction = generate_prediction(features)
        
        delay_risk = prediction['delay_risk']
        risk_label = ["Low", "Medium", "High"][delay_risk]
        
        print(f"   Delay Risk: {risk_label} (confidence: {prediction['confidence']:.2f})")
        print(f"   Yard Utilization: {features['yard_utilization_ratio']:.2%}")
        print(f"   Truck Wait: {features['avg_truck_wait_min']:.1f} min")
        print(f"   Weather: {weather_source} | Wind: {features['wind_speed']:.1f} km/h | Temp: {features['temperature']:.1f}°C")
        
        # Save prediction to database
        with get_db_session() as session:
            prediction_obj = save_prediction(
                session=session,
                timestamp=datetime.utcnow(),
                time_window_start=datetime.utcnow(),
                predicted_risk_class=delay_risk,
                probability_low=prediction['probabilities']['low'],
                probability_medium=prediction['probabilities']['medium'],
                probability_high=prediction['probabilities']['high'],
                model_type="RandomForest" if MODEL_AVAILABLE else "Simulated",
                yard_utilization_ratio=features['yard_utilization_ratio'],
                avg_truck_wait_min=features['avg_truck_wait_min']
            )
            print(f"   ✅ Prediction saved (ID: {prediction_obj.id})")
        
        # Generate optimization recommendations (every 3rd cycle to avoid too many runs)
        if random.random() < 0.33:  # 33% chance each cycle
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating optimization...")
            current_risk = float(delay_risk) + prediction['probabilities']['high']
            optimization = generate_optimization_recommendations(features, current_risk)
            
            print(f"   Recommended Cranes: {optimization['recommended_cranes']}")
            print(f"   Target Yard Util: {optimization['yard_utilization_target']:.2%}")
            print(f"   Improvement: {optimization['improvement_percent']:.1f}%")
            
            # Save optimization to database
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            with get_db_session() as session:
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
                print(f"   ✅ Optimization saved (Run ID: {run_id})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function: Run continuous data ingestion
    """
    print("=" * 70)
    print("LIVE DATA INGESTION FOR POWER BI")
    print("=" * 70)
    print("\nThis script continuously writes predictions and optimization results")
    print("to PostgreSQL tables for Power BI consumption.")
    print("\nPress Ctrl+C to stop.")
    print("=" * 70)
    
    # Configuration
    INGESTION_INTERVAL_SECONDS = 300  # 5 minutes (adjust as needed)
    
    # Test database connection
    print("\n[1/3] Testing database connection...")
    if not test_connection():
        print("[ERROR] Cannot connect to database!")
        print("Please ensure PostgreSQL is running and credentials are correct.")
        return
    
    print("[SUCCESS] Database connected!")
    
    # Check model availability
    print("\n[2/3] Checking ML model availability...")
    if MODEL_AVAILABLE and MODEL_LOADED is not None:
        print(f"[SUCCESS] ML model loaded: {type(MODEL_LOADED).__name__}")
        print(f"   Features: {len(MODEL_FEATURES)} features available")
    else:
        print("[INFO] ML model not found - using simulated predictions")
        print("   ✅ This is OK - script will work perfectly with simulated data")
        print("   📝 To use real ML model: Train model and save to output/models/")
    
    # Test weather API
    print("\n[2.5/3] Testing weather API...")
    test_weather = fetch_weather_data()
    if test_weather:
        print(f"[SUCCESS] Weather API connected!")
        print(f"   Current weather: {test_weather.get('weather_main', 'Unknown')}")
        print(f"   Temperature: {test_weather.get('temperature', 0):.1f}°C")
        print(f"   Wind Speed: {test_weather.get('wind_speed', 0):.1f} km/h")
    else:
        print("[INFO] Weather API not available - using simulated weather data")
        print("   ✅ This is OK - script will work with simulated weather")
    
    # Start ingestion loop
    print("\n[3/3] Starting data ingestion...")
    print(f"   Interval: {INGESTION_INTERVAL_SECONDS} seconds ({INGESTION_INTERVAL_SECONDS/60:.1f} minutes)")
    print(f"   Database: port_congestion_db")
    print(f"   Tables: predictions, optimization_runs, optimization_recommendations")
    print("\n" + "=" * 70)
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n--- Cycle #{cycle_count} ---")
            
            success = ingest_data_cycle()
            
            if success:
                print(f"\n[SUCCESS] Cycle #{cycle_count} completed. Waiting {INGESTION_INTERVAL_SECONDS} seconds...")
            else:
                print(f"\n[WARNING] Cycle #{cycle_count} had errors. Retrying in {INGESTION_INTERVAL_SECONDS} seconds...")
            
            time.sleep(INGESTION_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(f"[STOPPED] Data ingestion stopped after {cycle_count} cycles.")
        print("=" * 70)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

