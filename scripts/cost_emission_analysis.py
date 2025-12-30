"""
Cost and Emission Analysis Module
Calculates delay costs and CO2 emissions based on predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
COST_EMISSION_FILE = PROJECT_DIR / "cost_emission.csv"
VESSEL_FILE = PROJECT_DIR / "vessel_port_calls.csv"


def load_cost_emission_data() -> pd.DataFrame:
    """Load cost and emission data"""
    if not COST_EMISSION_FILE.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(COST_EMISSION_FILE, low_memory=False)
    return df


def calculate_delay_cost(
    vessel_id: str,
    delay_hours: float,
    fuel_cost_per_hour: Optional[float] = None,
    base_cost_data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate delay cost for a vessel
    
    Args:
        vessel_id: Vessel identifier
        delay_hours: Hours of delay
        fuel_cost_per_hour: Fuel cost per hour (if known)
        base_cost_data: DataFrame with cost data
        
    Returns:
        Dictionary with cost breakdown
    """
    # Load cost data if not provided
    if base_cost_data is None:
        base_cost_data = load_cost_emission_data()
    
    # Get vessel-specific cost if available
    if not base_cost_data.empty and vessel_id in base_cost_data['vessel_id'].values:
        vessel_data = base_cost_data[base_cost_data['vessel_id'] == vessel_id].iloc[0]
        fuel_cost_per_hour = vessel_data.get('fuel_cost_per_hour_usd', 3000)
        co2_per_hour = vessel_data.get('co2_emission_kg_per_hour', 500)
    else:
        # Default values
        fuel_cost_per_hour = fuel_cost_per_hour or 3000.0
        co2_per_hour = 500.0
    
    # Calculate costs
    fuel_cost = delay_hours * fuel_cost_per_hour
    
    # Additional costs (penalties, opportunity cost, etc.)
    penalty_cost = max(0, (delay_hours - 2) * 5000) if delay_hours > 2 else 0
    opportunity_cost = delay_hours * 2000  # Estimated opportunity cost
    
    total_cost = fuel_cost + penalty_cost + opportunity_cost
    
    # Calculate emissions
    co2_emission_kg = delay_hours * co2_per_hour
    co2_emission_tonnes = co2_emission_kg / 1000
    
    return {
        'delay_hours': delay_hours,
        'fuel_cost_usd': fuel_cost,
        'penalty_cost_usd': penalty_cost,
        'opportunity_cost_usd': opportunity_cost,
        'total_delay_cost_usd': total_cost,
        'co2_emission_kg': co2_emission_kg,
        'co2_emission_tonnes': co2_emission_tonnes,
        'fuel_cost_per_hour_usd': fuel_cost_per_hour,
        'co2_per_hour_kg': co2_per_hour
    }


def calculate_prediction_cost_impact(
    prediction_data: pd.DataFrame,
    cost_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Calculate cost and emission impact for predictions
    
    Args:
        prediction_data: DataFrame with predictions (must have delay_risk or delay_hours)
        cost_data: Optional cost data DataFrame
        
    Returns:
        DataFrame with cost and emission calculations
    """
    result = prediction_data.copy()
    
    # Map delay risk to estimated delay hours
    risk_to_hours = {
        0: 0.25,  # Low risk: 15 minutes average delay
        1: 1.25,  # Medium risk: 1.25 hours average delay
        2: 3.0    # High risk: 3 hours average delay
    }
    
    # Calculate delay hours from risk class
    if 'predicted_risk_class' in result.columns:
        result['estimated_delay_hours'] = result['predicted_risk_class'].map(risk_to_hours)
    elif 'delay_risk_24h' in result.columns:
        result['estimated_delay_hours'] = result['delay_risk_24h'].map(risk_to_hours)
    elif 'delay_hours' in result.columns:
        result['estimated_delay_hours'] = result['delay_hours']
    else:
        result['estimated_delay_hours'] = 1.0  # Default
    
    # Get vessel IDs if available
    vessel_ids = None
    if 'vessel_id' in result.columns:
        vessel_ids = result['vessel_id'].values
    elif cost_data is not None and not cost_data.empty:
        # Try to match by timestamp or other means
        vessel_ids = None
    
    # Calculate costs for each prediction
    cost_results = []
    for idx, row in result.iterrows():
        vessel_id = vessel_ids[idx] if vessel_ids is not None else f"VSL{idx:06d}"
        delay_hours = row['estimated_delay_hours']
        
        cost_info = calculate_delay_cost(
            vessel_id=str(vessel_id),
            delay_hours=float(delay_hours),
            base_cost_data=cost_data
        )
        cost_results.append(cost_info)
    
    # Merge cost results
    cost_df = pd.DataFrame(cost_results)
    result = pd.concat([result, cost_df], axis=1)
    
    return result


def aggregate_cost_emission_summary(
    cost_analysis_df: pd.DataFrame,
    group_by: Optional[str] = None
) -> Dict:
    """
    Aggregate cost and emission summary
    
    Args:
        cost_analysis_df: DataFrame with cost/emission calculations
        group_by: Optional column to group by (e.g., 'predicted_risk_class')
        
    Returns:
        Dictionary with aggregated metrics
    """
    if group_by and group_by in cost_analysis_df.columns:
        grouped = cost_analysis_df.groupby(group_by)
        summary = {
            'total_delay_cost_usd': grouped['total_delay_cost_usd'].sum().to_dict(),
            'total_co2_emission_tonnes': grouped['co2_emission_tonnes'].sum().to_dict(),
            'avg_delay_hours': grouped['estimated_delay_hours'].mean().to_dict(),
            'vessel_count': grouped.size().to_dict()
        }
    else:
        summary = {
            'total_delay_cost_usd': float(cost_analysis_df['total_delay_cost_usd'].sum()),
            'total_co2_emission_tonnes': float(cost_analysis_df['co2_emission_tonnes'].sum()),
            'avg_delay_hours': float(cost_analysis_df['estimated_delay_hours'].mean()),
            'vessel_count': len(cost_analysis_df),
            'avg_cost_per_vessel_usd': float(cost_analysis_df['total_delay_cost_usd'].mean()),
            'avg_co2_per_vessel_tonnes': float(cost_analysis_df['co2_emission_tonnes'].mean())
        }
    
    return summary


def main():
    """Test cost and emission calculations"""
    print("=" * 70)
    print("COST & EMISSION ANALYSIS MODULE")
    print("=" * 70)
    
    # Load cost data
    cost_data = load_cost_emission_data()
    if not cost_data.empty:
        print(f"Loaded {len(cost_data):,} cost/emission records")
    else:
        print("Cost data file not found, using default values")
    
    # Test calculation
    test_vessel = "VSL100000"
    test_delay = 2.5
    
    result = calculate_delay_cost(test_vessel, test_delay, base_cost_data=cost_data)
    
    print(f"\nTest Calculation for Vessel {test_vessel}:")
    print(f"  Delay: {test_delay} hours")
    print(f"  Fuel Cost: ${result['fuel_cost_usd']:,.2f}")
    print(f"  Penalty Cost: ${result['penalty_cost_usd']:,.2f}")
    print(f"  Opportunity Cost: ${result['opportunity_cost_usd']:,.2f}")
    print(f"  Total Cost: ${result['total_delay_cost_usd']:,.2f}")
    print(f"  CO2 Emission: {result['co2_emission_tonnes']:.2f} tonnes")
    
    print("\n[SUCCESS] Cost & Emission Analysis Module Ready!")


if __name__ == "__main__":
    main()

