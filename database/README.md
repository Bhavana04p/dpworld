# PostgreSQL Database Integration - Complete Setup Guide

## Overview

This module integrates PostgreSQL database to store:
- **Predictions**: ML model predictions for delay risk
- **Optimization Recommendations**: Resource allocation recommendations
- **Operational Decisions**: Actual decisions made based on recommendations
- **Optimization Runs**: Metadata about each optimization execution

## Prerequisites

1. **PostgreSQL installed** (version 12 or higher)
2. **Python packages**: `psycopg2` or `psycopg2-binary`, `sqlalchemy`
3. **Database created** (we'll create it in setup)

## Step 1: Install PostgreSQL

### Windows
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer
3. Remember the password you set for `postgres` user
4. Default port: 5432

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### macOS
```bash
brew install postgresql
brew services start postgresql
```

## Step 2: Create Database

### Option A: Using psql (Command Line)

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE port_congestion_db;

# Create user (optional, or use postgres)
CREATE USER port_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE port_congestion_db TO port_user;

# Exit
\q
```

### Option B: Using pgAdmin (GUI)

1. Open pgAdmin
2. Right-click on "Databases" → "Create" → "Database"
3. Name: `port_congestion_db`
4. Click "Save"

## Step 3: Install Python Dependencies

```bash
pip install psycopg2-binary sqlalchemy
```

Or add to `requirements.txt`:
```
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
```

## Step 4: Configure Database Connection

### Option A: Environment Variables (Recommended)

Create a `.env` file in project root:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=port_congestion_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Or set in your system:
```bash
# Windows PowerShell
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="port_congestion_db"
$env:DB_USER="postgres"
$env:DB_PASSWORD="your_password"

# Linux/macOS
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=port_congestion_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Option B: Edit `database/db_connection.py`

Modify the constants at the top:
```python
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'port_congestion_db'
DB_USER = 'postgres'
DB_PASSWORD = 'your_password'
```

## Step 5: Initialize Database Tables

Run the initialization script:

```bash
python database/init_db.py
```

This will:
1. Test database connection
2. Create all required tables:
   - `predictions`
   - `optimization_recommendations`
   - `operational_decisions`
   - `optimization_runs`

## Step 6: Verify Setup

Test the connection:

```python
from database.db_connection import test_connection, get_database_info

# Test connection
if test_connection():
    print("✅ Database connected!")
    
    # Get info
    info = get_database_info()
    print(f"Tables: {info['tables']}")
else:
    print("❌ Connection failed")
```

## Database Schema

### Tables Created

1. **predictions**
   - Stores ML model predictions
   - Links to recommendations
   - Indexed by timestamp and risk class

2. **optimization_recommendations**
   - Stores resource allocation recommendations
   - Links to optimization runs
   - Indexed by run_id and start_time

3. **operational_decisions**
   - Stores actual decisions made
   - Links to recommendations
   - Tracks implementation status

4. **optimization_runs**
   - Stores metadata about optimization executions
   - Contains configuration and results summary
   - Indexed by run_timestamp

## Usage Examples

### Save a Prediction

```python
from database.db_connection import get_db_session
from database.db_operations import save_prediction
from datetime import datetime

with get_db_session() as session:
    prediction = save_prediction(
        session=session,
        timestamp=datetime.utcnow(),
        time_window_start=datetime(2023, 1, 1, 0, 0),
        predicted_risk_class=1,  # Medium
        probability_low=0.2,
        probability_medium=0.6,
        probability_high=0.2,
        model_type="RandomForest",
        yard_utilization_ratio=0.85,
        avg_truck_wait_min=45.5
    )
    print(f"Saved prediction ID: {prediction.id}")
```

### Save Optimization Results

The optimization script automatically saves to database when run:
```bash
python scripts/optimize_resources.py
```

### Query Predictions

```python
from database.db_connection import get_db_session
from database.db_operations import get_latest_predictions, get_predictions_by_date_range
from datetime import datetime, timedelta

with get_db_session() as session:
    # Get latest 100 predictions
    predictions = get_latest_predictions(session, limit=100)
    
    # Get predictions for last week
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    predictions = get_predictions_by_date_range(session, start_date, end_date)
```

### Query Optimization Recommendations

```python
from database.db_connection import get_db_session
from database.db_operations import (
    get_latest_optimization_run,
    get_optimization_recommendations_by_run,
    get_unimplemented_recommendations
)

with get_db_session() as session:
    # Get latest optimization run
    latest_run = get_latest_optimization_run(session)
    if latest_run:
        print(f"Latest run: {latest_run.run_id}")
        
        # Get all recommendations from this run
        recommendations = get_optimization_recommendations_by_run(
            session, latest_run.run_id
        )
        print(f"Found {len(recommendations)} recommendations")
    
    # Get unimplemented recommendations
    unimplemented = get_unimplemented_recommendations(session)
    print(f"Unimplemented: {len(unimplemented)}")
```

## Integration with Existing Code

### Optimization Script
The `scripts/optimize_resources.py` automatically saves to database when:
- Database connection is available
- Tables are initialized

### Dashboard
The dashboard can be updated to read from database (see next section)

## Troubleshooting

### Connection Refused
- Check PostgreSQL is running: `pg_isready` or check service status
- Verify host and port are correct
- Check firewall settings

### Authentication Failed
- Verify username and password
- Check pg_hba.conf for authentication method
- Ensure user has permissions

### Database Does Not Exist
- Create database: `CREATE DATABASE port_congestion_db;`
- Or run: `createdb -U postgres port_congestion_db`

### Module Not Found
- Install dependencies: `pip install psycopg2-binary sqlalchemy`
- Check Python path includes project directory

## Next Steps

1. Run optimization to populate database
2. Update dashboard to read from database
3. Create queries for analytics
4. Set up scheduled backups

