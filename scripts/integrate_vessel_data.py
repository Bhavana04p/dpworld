"""
Integrate vessel_port_calls.csv and AIS data into ML pipeline
Adds vessel-specific features for better predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_DIR = Path(__file__).parent.parent
VESSEL_FILE = PROJECT_DIR / "vessel_port_calls.csv"
AIS_FILE = PROJECT_DIR / "ais_tracking.csv"
OUTPUT_DIR = PROJECT_DIR / "output" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_process_vessel_data() -> pd.DataFrame:
    """Load and process vessel port calls data"""
    print("Loading vessel port calls data...")
    df = pd.read_csv(VESSEL_FILE, low_memory=False)
    
    # Convert datetime columns (without timezone)
    datetime_cols = ['scheduled_arrival', 'actual_arrival', 'departure_time']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=False)
    
    # Calculate vessel-specific features
    df['arrival_delay_hours'] = (df['actual_arrival'] - df['scheduled_arrival']).dt.total_seconds() / 3600
    df['turnaround_hours'] = (df['departure_time'] - df['actual_arrival']).dt.total_seconds() / 3600
    df['vessel_size_category'] = pd.cut(
        df['gross_tonnage'],
        bins=[0, 50000, 100000, 200000, np.inf],
        labels=['Small', 'Medium', 'Large', 'Very Large']
    )
    df['vessel_length_category'] = pd.cut(
        df['length_overall_m'],
        bins=[0, 200, 300, 400, np.inf],
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    # One-hot encode vessel type
    vessel_type_dummies = pd.get_dummies(df['vessel_type'], prefix='vessel_type')
    df = pd.concat([df, vessel_type_dummies], axis=1)
    
    # Aggregate by time window (hourly) - use 'h' instead of 'H'
    df['time_window'] = df['actual_arrival'].dt.floor('h')
    
    return df


def load_and_process_ais_data() -> pd.DataFrame:
    """Load and process AIS tracking data"""
    if not AIS_FILE.exists():
        print("AIS tracking file not found, skipping...")
        return pd.DataFrame()
    
    print("Loading AIS tracking data...")
    df = pd.read_csv(AIS_FILE, low_memory=False)
    
    # Convert timestamp (without timezone)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
    
    # Calculate ETA-based features
    df['eta_category'] = pd.cut(
        df['eta_hours'],
        bins=[0, 6, 12, 24, 48, np.inf],
        labels=['Immediate', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Speed categories
    df['speed_category'] = pd.cut(
        df['speed_knots'],
        bins=[0, 5, 10, 15, 20, np.inf],
        labels=['Docked', 'Slow', 'Medium', 'Fast', 'Very Fast']
    )
    
    # Aggregate by time window - use 'h' instead of 'H'
    if 'timestamp' in df.columns:
        df['time_window'] = df['timestamp'].dt.floor('h')
    
    return df


def merge_with_existing_dataset(vessel_df: pd.DataFrame, ais_df: pd.DataFrame) -> pd.DataFrame:
    """Merge vessel and AIS data with existing ML dataset"""
    # Load existing ML dataset
    ml_file = OUTPUT_DIR / "ml_features_targets_regression_refined.csv"
    if not ml_file.exists():
        ml_file = OUTPUT_DIR / "ml_features_targets_regression_refined.parquet"
    
    if not ml_file.exists():
        print("Existing ML dataset not found. Run feature engineering first.")
        return pd.DataFrame()
    
    print("Loading existing ML dataset...")
    if ml_file.suffix == '.parquet':
        ml_df = pd.read_parquet(ml_file)
    else:
        ml_df = pd.read_csv(ml_file, low_memory=False)
    
    # Find time column
    time_cols = [c for c in ml_df.columns if any(k in c.lower() for k in ['time', 'date', 'timestamp', 'ata', 'arrival'])]
    if not time_cols:
        print("No time column found in ML dataset")
        return ml_df
    
    time_col = time_cols[0]
    # Remove timezone if present
    ml_df[time_col] = pd.to_datetime(ml_df[time_col], errors='coerce')
    if ml_df[time_col].dt.tz is not None:
        ml_df[time_col] = ml_df[time_col].dt.tz_localize(None)
    ml_df['time_window'] = ml_df[time_col].dt.floor('h')
    
    # Ensure vessel_df time_window has no timezone
    if vessel_df['time_window'].dt.tz is not None:
        vessel_df['time_window'] = vessel_df['time_window'].dt.tz_localize(None)
    
    # Aggregate vessel data by time window
    vessel_agg = vessel_df.groupby('time_window').agg({
        'arrival_delay_hours': ['mean', 'std', 'count'],
        'turnaround_hours': ['mean', 'std'],
        'gross_tonnage': ['mean', 'sum'],
        'length_overall_m': ['mean', 'max'],
        'berth_id': 'nunique',
        'vessel_type_Container': 'sum',
        'vessel_type_Tanker': 'sum',
        'vessel_type_Bulk': 'sum'
    }).reset_index()
    
    # Flatten column names
    vessel_agg.columns = ['time_window'] + [f'vessel_{col[0]}_{col[1]}' if col[1] else f'vessel_{col[0]}' 
                                            for col in vessel_agg.columns[1:]]
    
    # Aggregate AIS data by time window
    if not ais_df.empty and 'time_window' in ais_df.columns:
        # Ensure AIS time_window has no timezone
        if ais_df['time_window'].dt.tz is not None:
            ais_df['time_window'] = ais_df['time_window'].dt.tz_localize(None)
        
        ais_agg = ais_df.groupby('time_window').agg({
            'speed_knots': ['mean', 'std'],
            'eta_hours': ['mean', 'min', 'max'],
            'mmsi': 'nunique'
        }).reset_index()
        
        ais_agg.columns = ['time_window'] + [f'ais_{col[0]}_{col[1]}' if col[1] else f'ais_{col[0]}' 
                                             for col in ais_agg.columns[1:]]
        
        # Merge AIS data
        ml_df = ml_df.merge(ais_agg, on='time_window', how='left')
    
    # Merge vessel data
    ml_df = ml_df.merge(vessel_agg, on='time_window', how='left')
    
    # Fill missing values
    vessel_cols = [c for c in ml_df.columns if c.startswith('vessel_')]
    ais_cols = [c for c in ml_df.columns if c.startswith('ais_')]
    
    for col in vessel_cols + ais_cols:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].fillna(0)
    
    # Drop time_window helper column
    if 'time_window' in ml_df.columns:
        ml_df = ml_df.drop(columns=['time_window'])
    
    return ml_df


def main():
    """Main function to integrate vessel data"""
    print("=" * 70)
    print("VESSEL DATA INTEGRATION")
    print("=" * 70)
    
    # Load vessel data
    vessel_df = load_and_process_vessel_data()
    print(f"Loaded {len(vessel_df):,} vessel records")
    
    # Load AIS data
    ais_df = load_and_process_ais_data()
    if not ais_df.empty:
        print(f"Loaded {len(ais_df):,} AIS records")
    
    # Merge with existing dataset
    enhanced_df = merge_with_existing_dataset(vessel_df, ais_df)
    
    if enhanced_df.empty:
        print("Failed to merge data")
        return
    
    # Save enhanced dataset
    output_file = OUTPUT_DIR / "ml_features_targets_regression_refined_with_vessels.csv"
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Enhanced dataset saved: {output_file}")
    print(f"   Original columns: {len(enhanced_df.columns) - len([c for c in enhanced_df.columns if c.startswith(('vessel_', 'ais_'))])}")
    print(f"   Vessel features added: {len([c for c in enhanced_df.columns if c.startswith('vessel_')])}")
    print(f"   AIS features added: {len([c for c in enhanced_df.columns if c.startswith('ais_')])}")
    print(f"   Total columns: {len(enhanced_df.columns)}")
    
    # Also save as parquet
    output_parquet = OUTPUT_DIR / "ml_features_targets_regression_refined_with_vessels.parquet"
    enhanced_df.to_parquet(output_parquet, index=False)
    print(f"   Also saved as: {output_parquet}")


if __name__ == "__main__":
    main()

