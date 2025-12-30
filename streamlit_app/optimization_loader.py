"""
Load and process optimization results for dashboard visualization
Supports both file-based and database-backed loading
"""
import os
import json
import glob
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Use Path for cross-platform compatibility
OPTIMIZATION_DIR = Path("output") / "optimization"

# Try to import database loader
try:
    from database.db_loader import (
        is_database_available,
        load_recommendations_from_db,
        load_optimization_summary_from_db,
        get_available_optimization_runs
    )
    DB_LOADER_AVAILABLE = True
except ImportError:
    DB_LOADER_AVAILABLE = False


def list_optimization_results() -> Dict[str, str]:
    """List all available optimization result files"""
    if not OPTIMIZATION_DIR.exists():
        return {}
    
    files = {}
    for pattern in ["recommendations_*.csv", "impact_analysis_*.json"]:
        for path in glob.glob(str(OPTIMIZATION_DIR / pattern)):
            basename = os.path.basename(path)
            files[basename] = path
    
    return dict(sorted(files.items(), reverse=True))  # Most recent first


def load_latest_recommendations() -> Optional[pd.DataFrame]:
    """Load the most recent recommendations (from database if available, else from files)"""
    # Try files first (usually more complete)
    files = list_optimization_results()
    recommendation_files = {k: v for k, v in files.items() if k.startswith("recommendations_")}
    
    file_recs = None
    if recommendation_files:
        try:
            latest_file = max(recommendation_files.values(), key=os.path.getmtime)
            file_recs = pd.read_csv(latest_file)
        except Exception as e:
            print(f"Error loading recommendations file: {e}")
    
    # Try database as fallback or if file has no data
    if DB_LOADER_AVAILABLE and is_database_available():
        try:
            db_recs = load_recommendations_from_db()
            if db_recs is not None and not db_recs.empty:
                # Use database if file is empty or database has more rows
                if file_recs is None or file_recs.empty:
                    return db_recs
                elif len(db_recs) > len(file_recs):
                    return db_recs
        except Exception:
            pass  # Fall through to file-based
    
    # Return file-based if available
    return file_recs


def load_latest_impact() -> Optional[Dict]:
    """Load the most recent impact analysis JSON"""
    files = list_optimization_results()
    impact_files = {k: v for k, v in files.items() if k.startswith("impact_analysis_")}
    
    if not impact_files:
        return None
    
    try:
        latest_file = max(impact_files.values(), key=os.path.getmtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading impact file: {e}")
        return None


def get_optimization_summary() -> Optional[Dict]:
    """Get summary of latest optimization run (from database if available, else from files)"""
    # Try database first
    if DB_LOADER_AVAILABLE and is_database_available():
        try:
            db_summary = load_optimization_summary_from_db()
            if db_summary:
                return db_summary
        except Exception:
            pass  # Fall through to file-based loading
    
    # Fallback to files
    impact = load_latest_impact()
    if not impact:
        return None
    
    # Handle different JSON structures
    # Check if impact has 'impact' key (nested) or direct keys
    if 'impact' in impact:
        impact_data = impact['impact']
    else:
        impact_data = impact
    
    # Extract before/after/improvements
    before = impact_data.get('before', {})
    after = impact_data.get('after', {})
    improvements = impact_data.get('improvements', {})
    
    # If structure is different, try to extract from top level
    if not before and 'before' in impact:
        before = impact.get('before', {})
    if not after and 'after' in impact:
        after = impact.get('after', {})
    if not improvements and 'improvements' in impact:
        improvements = impact.get('improvements', {})
    
    return {
        'status': impact.get('optimization_status', impact.get('status', 'unknown')),
        'timestamp': impact.get('timestamp', 'unknown'),
        'before': before,
        'after': after,
        'improvements': improvements,
        'config': impact.get('config', {})
    }


def prepare_comparison_data(recommendations_df: pd.DataFrame, impact: Dict) -> pd.DataFrame:
    """Prepare before/after comparison dataframe"""
    if recommendations_df is None or impact is None:
        return pd.DataFrame()
    
    comparison_data = []
    
    for _, row in recommendations_df.iterrows():
        comparison_data.append({
            'Time Window': f"Window {int(row.get('window_id', 0))}",
            'Start Time': row.get('start_time', ''),
            'Current Yard Util': row.get('current_yard_util', 0),
            'Recommended Yard Util': row.get('recommended_yard_util', 0),
            'Current Delay Risk': row.get('current_delay_risk', 0),
            'Expected Risk': row.get('current_delay_risk', 0) - row.get('expected_risk_reduction', 0),
            'Risk Reduction': row.get('expected_risk_reduction', 0),
            'Recommended Cranes': row.get('recommended_cranes', 0)
        })
    
    return pd.DataFrame(comparison_data)


def get_top_recommendations(recommendations_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Get top N recommendations by expected impact"""
    if recommendations_df is None or recommendations_df.empty:
        return pd.DataFrame()
    
    # Sort by risk reduction
    sorted_df = recommendations_df.sort_values('expected_risk_reduction', ascending=False)
    return sorted_df.head(top_n)

