"""
Quick test script to verify PostgreSQL connection
"""
from database.db_connection import test_connection, get_database_info

print("=" * 60)
print("POSTGRESQL CONNECTION TEST")
print("=" * 60)

print("\n[1/2] Testing connection...")
result = test_connection()

if result:
    print("\n[2/2] Getting database information...")
    info = get_database_info()
    
    print("\n" + "=" * 60)
    print("CONNECTION SUCCESSFUL!")
    print("=" * 60)
    print(f"Host: {info.get('host', 'N/A')}")
    print(f"Port: {info.get('port', 'N/A')}")
    print(f"Database: {info.get('database', 'N/A')}")
    print(f"User: {info.get('user', 'N/A')}")
    print(f"Connected: {info.get('connected', False)}")
    print(f"\nTables ({len(info.get('tables', []))}):")
    for table in info.get('tables', []):
        print(f"  - {table}")
    print("\n[SUCCESS] Database is ready to use!")
else:
    print("\n[ERROR] Connection failed!")
    print("Please check:")
    print("  1. PostgreSQL is running")
    print("  2. Database 'port_congestion_db' exists")
    print("  3. Password is correct in database/db_connection.py")

