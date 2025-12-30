"""
Direct PostgreSQL connection using psycopg2 (without SQLAlchemy dependency for basic operations)
Provides alternative connection method
"""
import os
from typing import Optional, Dict, List, Tuple
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

# Database configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'port_congestion_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Spoorthi@2005')


def get_connection_string() -> str:
    """Get PostgreSQL connection string"""
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


def test_direct_connection() -> Tuple[bool, str]:
    """
    Test direct PostgreSQL connection using psycopg2
    
    Returns:
        (success: bool, message: str)
    """
    if not PSYCOPG2_AVAILABLE:
        return False, "psycopg2 not installed"
    
    try:
        conn = psycopg2.connect(get_connection_string())
        conn.close()
        return True, "Connection successful"
    except psycopg2.OperationalError as e:
        return False, f"Connection failed: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict]:
    """
    Execute a SELECT query and return results
    
    Args:
        query: SQL query string
        params: Query parameters
    
    Returns:
        List of dictionaries (rows)
    """
    if not PSYCOPG2_AVAILABLE:
        return []
    
    try:
        conn = psycopg2.connect(get_connection_string())
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"Query error: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_table_info() -> Dict:
    """Get information about database tables"""
    if not PSYCOPG2_AVAILABLE:
        return {'connected': False, 'tables': []}
    
    try:
        conn = psycopg2.connect(get_connection_string())
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]
        
        conn.close()
        return {'connected': True, 'tables': tables}
    except Exception as e:
        return {'connected': False, 'error': str(e), 'tables': []}

