"""
Weather Data Integration Module
Uses weather_data.csv instead of API for historical weather
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

PROJECT_DIR = Path(__file__).parent.parent
WEATHER_FILE = PROJECT_DIR / "weather_data.csv"


def load_weather_data() -> pd.DataFrame:
    """Load historical weather data from CSV"""
    if not WEATHER_FILE.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(WEATHER_FILE, low_memory=False)
    
    # Convert timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df


def get_weather_for_timestamp(target_timestamp: datetime, weather_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Get weather data for a specific timestamp
    
    Args:
        target_timestamp: Target datetime
        weather_df: Optional pre-loaded weather DataFrame
        
    Returns:
        Dictionary with weather data or None
    """
    if weather_df is None:
        weather_df = load_weather_data()
    
    if weather_df.empty:
        return None
    
    # Find closest timestamp
    weather_df['time_diff'] = abs(weather_df['timestamp'] - target_timestamp)
    closest_idx = weather_df['time_diff'].idxmin()
    
    if pd.isna(closest_idx):
        return None
    
    row = weather_df.loc[closest_idx]
    
    # Convert to standard format
    weather_info = {
        'wind_speed': float(row.get('wind_speed_mps', 0)) * 3.6,  # Convert m/s to km/h
        'wave_height': float(row.get('wave_height_m', 0)),
        'rainfall': float(row.get('rainfall_mm', 0)),
        'visibility': float(row.get('visibility_km', 10)),
        'temperature': float(row.get('temperature_c', 20)),
        'timestamp': row['timestamp']
    }
    
    return weather_info


def integrate_weather_into_features(features: Dict, target_timestamp: Optional[datetime] = None) -> Dict:
    """
    Integrate weather data into feature dictionary
    
    Args:
        features: Feature dictionary
        target_timestamp: Target timestamp (default: now)
        
    Returns:
        Updated feature dictionary with weather data
    """
    if target_timestamp is None:
        target_timestamp = datetime.now()
    
    weather_data = get_weather_for_timestamp(target_timestamp)
    
    if weather_data:
        # Update features with real weather data
        features['wind_speed'] = weather_data['wind_speed']
        features['wave_height'] = weather_data['wave_height']
        features['rainfall'] = weather_data['rainfall']
        features['visibility'] = weather_data['visibility']
        features['temperature'] = weather_data['temperature']
        features['_weather_source'] = 'CSV'
    else:
        # Keep existing or use defaults
        features.setdefault('wind_speed', 15.0)
        features.setdefault('wave_height', 1.5)
        features.setdefault('rainfall', 0.0)
        features.setdefault('visibility', 10.0)
        features.setdefault('temperature', 25.0)
        features['_weather_source'] = 'Default'
    
    return features


def main():
    """Test weather data integration"""
    print("=" * 70)
    print("WEATHER DATA INTEGRATION MODULE")
    print("=" * 70)
    
    weather_df = load_weather_data()
    if not weather_df.empty:
        print(f"Loaded {len(weather_df):,} weather records")
        print(f"Date range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    else:
        print("Weather data file not found")
        return
    
    # Test getting weather for a specific timestamp
    test_timestamp = datetime(2023, 1, 1, 12, 0, 0)
    weather = get_weather_for_timestamp(test_timestamp, weather_df)
    
    if weather:
        print(f"\nWeather for {test_timestamp}:")
        print(f"  Wind Speed: {weather['wind_speed']:.1f} km/h")
        print(f"  Temperature: {weather['temperature']:.1f}°C")
        print(f"  Rainfall: {weather['rainfall']:.1f} mm")
        print(f"  Visibility: {weather['visibility']:.1f} km")
        print(f"  Wave Height: {weather['wave_height']:.2f} m")
    
    # Test feature integration
    test_features = {'yard_utilization_ratio': 0.85, 'avg_truck_wait_min': 45.0}
    enhanced_features = integrate_weather_into_features(test_features, test_timestamp)
    
    print(f"\nEnhanced features:")
    print(f"  Weather Source: {enhanced_features.get('_weather_source', 'Unknown')}")
    print(f"  Wind Speed: {enhanced_features.get('wind_speed', 0):.1f} km/h")
    print(f"  Temperature: {enhanced_features.get('temperature', 0):.1f}°C")
    
    print("\n[SUCCESS] Weather Data Integration Module Ready!")


if __name__ == "__main__":
    main()

