"""
Database connection utilities for PostgreSQL
Handles connection, session management, and initialization
"""
import os
from typing import Optional
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

from database.models import Base

# Database configuration - Direct PostgreSQL connection settings
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'port_congestion_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Spoorthi@2005')  # Your PostgreSQL password

# Connection string - URL encode password to handle special characters like @
# This prevents issues when password contains @, :, /, etc.
encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def create_database_engine(echo: bool = False):
    """
    Create SQLAlchemy engine
    
    Args:
        echo: If True, log all SQL statements
    
    Returns:
        SQLAlchemy engine
    """
    try:
        engine = create_engine(
            DATABASE_URL,
            echo=echo,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,
            max_overflow=10
        )
        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to create database engine: {str(e)}")


def get_session_factory(engine=None):
    """
    Get session factory
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
    
    Returns:
        Session factory
    """
    if engine is None:
        engine = create_database_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_db_session(engine=None, commit: bool = True):
    """
    Context manager for database sessions
    
    Usage:
        with get_db_session() as session:
            # Use session
            pass
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
        commit: Whether to commit on exit (True) or rollback (False)
    """
    if engine is None:
        engine = create_database_engine()
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        yield session
        if commit:
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def init_database(engine=None, drop_existing: bool = False):
    """
    Initialize database - create all tables
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
        drop_existing: If True, drop existing tables first (WARNING: deletes data!)
    
    Returns:
        True if successful
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        if drop_existing:
            print("WARNING: Dropping existing tables...")
            Base.metadata.drop_all(engine)
        
        print("Creating database tables...")
        Base.metadata.create_all(engine)
        print("Database tables created successfully!")
        return True
    except SQLAlchemyError as e:
        print(f"Error initializing database: {str(e)}")
        raise


def test_connection(engine=None) -> bool:
    """
    Test database connection
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
    
    Returns:
        True if connection successful
    """
    if engine is None:
        engine = create_database_engine()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False


def get_database_info(engine=None) -> dict:
    """
    Get database information
    
    Args:
        engine: SQLAlchemy engine (creates new if None)
    
    Returns:
        Dictionary with database info
    """
    if engine is None:
        engine = create_database_engine()
    
    info = {
        'host': DB_HOST,
        'port': DB_PORT,
        'database': DB_NAME,
        'user': DB_USER,
        'connected': False,
        'tables': []
    }
    
    try:
        with engine.connect() as conn:
            info['connected'] = True
            # Get table names
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            info['tables'] = [row[0] for row in result]
    except Exception as e:
        info['error'] = str(e)
    
    return info

