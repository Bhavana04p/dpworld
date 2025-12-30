"""
Database connection and session management for FastAPI
Reuses existing database connection utilities
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.db_connection import get_db_session, create_database_engine, test_connection
from database.db_operations import (
    save_prediction,
    save_optimization_run,
    save_optimization_recommendations
)
from sqlalchemy.orm import Session
from contextlib import contextmanager

# Re-export for convenience
__all__ = [
    'get_db_session',
    'create_database_engine',
    'test_connection',
    'save_prediction',
    'save_optimization_run',
    'save_optimization_recommendations',
    'get_db_session_for_api'
]


@contextmanager
def get_db_session_for_api():
    """
    Context manager for database sessions in API
    Automatically handles commit/rollback
    """
    with get_db_session() as session:
        yield session

