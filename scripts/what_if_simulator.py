"""
What-If Simulation Module
Interactive scenario planning for berth allocation and crane deployment
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

PROJECT_DIR = Path(__file__).parent.parent


def simulate_berth_allocation(
    current_state: Dict,
    proposed_allocation: Dict,
    time_horizon_hours: int = 24
) -> Dict:
    """
    Simulate impact of different berth allocation strategies
    
    Args:
        current_state: Current operational state
        proposed_allocation: Proposed berth allocation changes
        time_horizon_hours: Time horizon for simulation
        
    Returns:
        Dictionary with simulation results
    """
    # Extract current metrics
    current_yard_util = current_state.get('yard_utilization_ratio', 0.85)
    current_cranes = current_state.get('available_cranes', 20)
    current_delay_risk = current_state.get('current_delay_risk', 1.5)
    
    # Proposed changes
    proposed_cranes = proposed_allocation.get('cranes_per_berth', {})
    proposed_berth_priority = proposed_allocation.get('berth_priority', {})
    
    # Simulate impact
    total_proposed_cranes = sum(proposed_cranes.values()) if proposed_cranes else current_cranes
    
    # Calculate expected improvements
    crane_improvement_factor = min(1.5, total_proposed_cranes / max(1, current_cranes))
    yard_util_improvement = current_yard_util * (1 - (crane_improvement_factor - 1) * 0.1)
    yard_util_improvement = max(0.5, min(0.95, yard_util_improvement))
    
    # Estimate delay risk reduction
    risk_reduction = (current_yard_util - yard_util_improvement) * 2.0
    new_delay_risk = max(0.0, current_delay_risk - risk_reduction)
    
    # Calculate cost impact
    crane_cost_per_hour = 500  # USD per crane per hour
    additional_cranes = max(0, total_proposed_cranes - current_cranes)
    additional_cost = additional_cranes * crane_cost_per_hour * time_horizon_hours
    
    # Calculate benefit (reduced delay cost)
    delay_reduction_hours = (current_delay_risk - new_delay_risk) * 1.5
    delay_cost_saved = delay_reduction_hours * 3000  # USD per hour of delay
    
    net_benefit = delay_cost_saved - additional_cost
    
    return {
        'scenario': 'berth_allocation',
        'time_horizon_hours': time_horizon_hours,
        'current_state': {
            'yard_utilization': current_yard_util,
            'cranes': current_cranes,
            'delay_risk': current_delay_risk
        },
        'proposed_state': {
            'yard_utilization': yard_util_improvement,
            'cranes': total_proposed_cranes,
            'delay_risk': new_delay_risk
        },
        'improvements': {
            'yard_util_improvement_pct': (current_yard_util - yard_util_improvement) / current_yard_util * 100,
            'delay_risk_reduction': current_delay_risk - new_delay_risk,
            'delay_risk_reduction_pct': (current_delay_risk - new_delay_risk) / max(0.1, current_delay_risk) * 100
        },
        'cost_analysis': {
            'additional_crane_cost_usd': additional_cost,
            'delay_cost_saved_usd': delay_cost_saved,
            'net_benefit_usd': net_benefit,
            'roi_percent': (net_benefit / max(1, additional_cost)) * 100 if additional_cost > 0 else float('inf')
        },
        'recommendation': 'proceed' if net_benefit > 0 else 'reconsider'
    }


def simulate_crane_deployment(
    current_state: Dict,
    proposed_deployment: Dict,
    time_horizon_hours: int = 24
) -> Dict:
    """
    Simulate impact of different crane deployment strategies
    
    Args:
        current_state: Current operational state
        proposed_deployment: Proposed crane deployment changes
        time_horizon_hours: Time horizon for simulation
        
    Returns:
        Dictionary with simulation results
    """
    current_cranes = current_state.get('available_cranes', 20)
    current_yard_util = current_state.get('yard_utilization_ratio', 0.85)
    current_delay_risk = current_state.get('current_delay_risk', 1.5)
    
    # Proposed deployment
    proposed_cranes = proposed_deployment.get('total_cranes', current_cranes)
    crane_distribution = proposed_deployment.get('distribution', {})
    
    # Calculate impact
    crane_change = proposed_cranes - current_cranes
    crane_change_pct = crane_change / max(1, current_cranes) * 100
    
    # Yard utilization improvement
    if crane_change > 0:
        yard_util_improvement = current_yard_util * (1 - abs(crane_change_pct) * 0.01)
    else:
        yard_util_improvement = current_yard_util * (1 + abs(crane_change_pct) * 0.01)
    
    yard_util_improvement = max(0.5, min(0.95, yard_util_improvement))
    
    # Delay risk impact
    util_change = current_yard_util - yard_util_improvement
    risk_change = util_change * 2.0
    new_delay_risk = max(0.0, current_delay_risk - risk_change)
    
    # Cost analysis
    crane_cost_per_hour = 500
    if crane_change > 0:
        additional_cost = crane_change * crane_cost_per_hour * time_horizon_hours
    else:
        additional_cost = 0  # Cost savings not calculated here
    
    delay_reduction = current_delay_risk - new_delay_risk
    delay_cost_saved = delay_reduction * 1.5 * 3000 * time_horizon_hours / 24
    
    net_benefit = delay_cost_saved - additional_cost
    
    return {
        'scenario': 'crane_deployment',
        'time_horizon_hours': time_horizon_hours,
        'current_state': {
            'cranes': current_cranes,
            'yard_utilization': current_yard_util,
            'delay_risk': current_delay_risk
        },
        'proposed_state': {
            'cranes': proposed_cranes,
            'yard_utilization': yard_util_improvement,
            'delay_risk': new_delay_risk
        },
        'improvements': {
            'crane_change': crane_change,
            'yard_util_improvement_pct': (current_yard_util - yard_util_improvement) / current_yard_util * 100,
            'delay_risk_reduction': delay_reduction,
            'delay_risk_reduction_pct': delay_reduction / max(0.1, current_delay_risk) * 100
        },
        'cost_analysis': {
            'additional_crane_cost_usd': additional_cost,
            'delay_cost_saved_usd': delay_cost_saved,
            'net_benefit_usd': net_benefit,
            'roi_percent': (net_benefit / max(1, additional_cost)) * 100 if additional_cost > 0 else float('inf')
        },
        'recommendation': 'proceed' if net_benefit > 0 else 'reconsider'
    }


def compare_scenarios(scenarios: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple what-if scenarios
    
    Args:
        scenarios: List of scenario results
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for scenario in scenarios:
        comparison_data.append({
            'scenario_name': scenario.get('scenario', 'unknown'),
            'delay_risk_before': scenario['current_state']['delay_risk'],
            'delay_risk_after': scenario['proposed_state']['delay_risk'],
            'risk_reduction_pct': scenario['improvements']['delay_risk_reduction_pct'],
            'yard_util_before': scenario['current_state']['yard_utilization'],
            'yard_util_after': scenario['proposed_state']['yard_utilization'],
            'net_benefit_usd': scenario['cost_analysis']['net_benefit_usd'],
            'roi_percent': scenario['cost_analysis']['roi_percent'],
            'recommendation': scenario['recommendation']
        })
    
    return pd.DataFrame(comparison_data)


def main():
    """Test what-if simulation"""
    print("=" * 70)
    print("WHAT-IF SIMULATION MODULE")
    print("=" * 70)
    
    # Test berth allocation scenario
    current_state = {
        'yard_utilization_ratio': 0.85,
        'available_cranes': 20,
        'current_delay_risk': 1.5
    }
    
    proposed_allocation = {
        'cranes_per_berth': {1: 3, 2: 3, 3: 2},
        'berth_priority': {1: 'high', 2: 'high', 3: 'medium'}
    }
    
    print("\n[1] Testing Berth Allocation Scenario...")
    berth_result = simulate_berth_allocation(current_state, proposed_allocation)
    print(f"   Delay Risk: {berth_result['current_state']['delay_risk']:.2f} -> {berth_result['proposed_state']['delay_risk']:.2f}")
    print(f"   Net Benefit: ${berth_result['cost_analysis']['net_benefit_usd']:,.2f}")
    print(f"   Recommendation: {berth_result['recommendation']}")
    
    # Test crane deployment scenario
    proposed_deployment = {
        'total_cranes': 25,
        'distribution': {'peak_hours': 25, 'off_hours': 15}
    }
    
    print("\n[2] Testing Crane Deployment Scenario...")
    crane_result = simulate_crane_deployment(current_state, proposed_deployment)
    print(f"   Delay Risk: {crane_result['current_state']['delay_risk']:.2f} -> {crane_result['proposed_state']['delay_risk']:.2f}")
    print(f"   Net Benefit: ${crane_result['cost_analysis']['net_benefit_usd']:,.2f}")
    print(f"   Recommendation: {crane_result['recommendation']}")
    
    # Compare scenarios
    print("\n[3] Scenario Comparison...")
    comparison = compare_scenarios([berth_result, crane_result])
    print(comparison.to_string(index=False))
    
    print("\n[SUCCESS] What-If Simulation Module Ready!")


if __name__ == "__main__":
    main()

