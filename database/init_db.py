"""
Initialize PostgreSQL database for Port Congestion System
Creates database and all tables
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_connection import init_database, test_connection, get_database_info, create_database_engine
from database.models import Base


def main():
    """Main initialization function"""
    print("=" * 60)
    print("PORT CONGESTION DATABASE INITIALIZATION")
    print("=" * 60)
    
    # Test connection first
    print("\n[1/3] Testing database connection...")
    if not test_connection():
        print("\n❌ Connection failed. Please check:")
        print("   1. PostgreSQL is running")
        print("   2. Database credentials are correct")
        print("   3. Database exists (create it if needed)")
        print("\nTo create database:")
        print("   psql -U postgres")
        print("   CREATE DATABASE port_congestion_db;")
        return False
    
    # Get database info
    print("\n[2/3] Database Information:")
    info = get_database_info()
    print(f"   Host: {info['host']}")
    print(f"   Port: {info['port']}")
    print(f"   Database: {info['database']}")
    print(f"   User: {info['user']}")
    print(f"   Connected: {info['connected']}")
    if info.get('tables'):
        print(f"   Existing tables: {', '.join(info['tables'])}")
    
    # Initialize database
    print("\n[3/3] Initializing database tables...")
    try:
        # Check if running in interactive mode
        import sys
        if sys.stdin.isatty():
            drop_existing = input("\nDrop existing tables? (y/N): ").lower() == 'y'
        else:
            # Non-interactive mode - don't drop existing
            drop_existing = False
            print("   Running in non-interactive mode - keeping existing tables")
        
        init_database(drop_existing=drop_existing)
        print("\n[SUCCESS] Database initialized successfully!")
        
        # Show created tables
        info = get_database_info()
        if info.get('tables'):
            print(f"\nCreated tables: {', '.join(info['tables'])}")
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Error initializing database: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

