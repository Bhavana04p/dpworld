"""
Cost and Emission Service
Calculates delay costs and CO2 emissions
"""
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.cost_emission_analysis import (
    calculate_delay_cost,
    calculate_prediction_cost_impact,
    aggregate_cost_emission_summary
)


def calculate_prediction_costs(prediction_data: Dict, vessel_id: Optional[str] = None) -> Dict:
    """
    Calculate cost and emission impact for a prediction
    
    Args:
        prediction_data: Prediction result dictionary
        vessel_id: Optional vessel ID
        
    Returns:
        Dictionary with cost and emission calculations
    """
    # Map delay risk to estimated delay hours
    risk_to_hours = {
        0: 0.25,  # Low risk: 15 minutes average delay
        1: 1.25,  # Medium risk: 1.25 hours average delay
        2: 3.0    # High risk: 3 hours average delay
    }
    
    delay_risk = prediction_data.get('delay_risk', prediction_data.get('predicted_risk_class', 1))
    delay_hours = risk_to_hours.get(delay_risk, 1.25)
    
    # Calculate costs
    cost_info = calculate_delay_cost(
        vessel_id=vessel_id or "VSL000000",
        delay_hours=delay_hours
    )
    
    return {
        'prediction': prediction_data,
        'cost_analysis': cost_info,
        'estimated_delay_hours': delay_hours
    }


def calculate_optimization_cost_impact(
    before_state: Dict,
    after_state: Dict,
    time_horizon_hours: int = 24
) -> Dict:
    """
    Calculate cost impact of optimization
    
    Args:
        before_state: State before optimization
        after_state: State after optimization
        time_horizon_hours: Time horizon
        
    Returns:
        Dictionary with cost impact analysis
    """
    before_risk = before_state.get('delay_risk', 1.5)
    after_risk = after_state.get('delay_risk', 0.8)
    
    risk_to_hours = {0: 0.25, 1: 1.25, 2: 3.0}
    
    before_hours = risk_to_hours.get(int(before_risk), before_risk * 1.5)
    after_hours = risk_to_hours.get(int(after_risk), after_risk * 1.5)
    
    # Calculate costs
    before_cost = calculate_delay_cost("VSL000000", before_hours * time_horizon_hours / 24)
    after_cost = calculate_delay_cost("VSL000000", after_hours * time_horizon_hours / 24)
    
    cost_saved = before_cost['total_delay_cost_usd'] - after_cost['total_delay_cost_usd']
    co2_saved = before_cost['co2_emission_tonnes'] - after_cost['co2_emission_tonnes']
    
    return {
        'before': {
            'delay_hours': before_hours,
            'total_cost_usd': before_cost['total_delay_cost_usd'],
            'co2_emission_tonnes': before_cost['co2_emission_tonnes']
        },
        'after': {
            'delay_hours': after_hours,
            'total_cost_usd': after_cost['total_delay_cost_usd'],
            'co2_emission_tonnes': after_cost['co2_emission_tonnes']
        },
        'savings': {
            'cost_saved_usd': cost_saved,
            'co2_saved_tonnes': co2_saved,
            'cost_reduction_pct': (cost_saved / max(0.1, before_cost['total_delay_cost_usd'])) * 100,
            'co2_reduction_pct': (co2_saved / max(0.1, before_cost['co2_emission_tonnes'])) * 100
        }
    }

