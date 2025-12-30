# PostgreSQL Database Setup - Complete Guide

## 🎯 What This Does

Integrates PostgreSQL database to store:
- ✅ ML predictions (delay risk forecasts)
- ✅ Optimization recommendations (crane allocation, yard utilization)
- ✅ Operational decisions (actual actions taken)
- ✅ Optimization run history (metadata and results)

## 📋 Step-by-Step Setup

### Step 1: Install PostgreSQL

#### Windows
1. Download installer: https://www.postgresql.org/download/windows/
2. Run installer, follow prompts
3. **Remember the password** you set for `postgres` user
4. Default port: **5432**
5. Installation location: Usually `C:\Program Files\PostgreSQL\15\` (version may vary)

#### Verify Installation
```bash
# Check if PostgreSQL is running
# Windows: Check Services (services.msc) for "postgresql-x64-15"
# Or use:
psql --version
```

### Step 2: Create Database

#### Option A: Using Command Line (psql)

```bash
# Connect to PostgreSQL
psql -U postgres

# Enter your password when prompted
# Then run:
CREATE DATABASE port_congestion_db;

# Verify it was created
\l

# Exit
\q
```

#### Option B: Using pgAdmin (GUI - Windows)

1. Open **pgAdmin** (installed with PostgreSQL)
2. Connect to server (enter postgres password)
3. Right-click **"Databases"** → **"Create"** → **"Database"**
4. Name: `port_congestion_db`
5. Click **"Save"**

### Step 3: Install Python Dependencies

```bash
pip install psycopg2-binary sqlalchemy
```

Or update requirements:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Database Connection

#### Method 1: Environment Variables (Recommended)

**Windows PowerShell:**
```powershell
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="port_congestion_db"
$env:DB_USER="postgres"
$env:DB_PASSWORD="your_postgres_password"
```

**Windows CMD:**
```cmd
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=port_congestion_db
set DB_USER=postgres
set DB_PASSWORD=your_postgres_password
```

**Linux/macOS:**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=port_congestion_db
export DB_USER=postgres
export DB_PASSWORD=your_postgres_password
```

#### Method 2: Edit `database/db_connection.py`

Open `database/db_connection.py` and modify:
```python
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'port_congestion_db'
DB_USER = 'postgres'
DB_PASSWORD = 'your_postgres_password'  # Change this!
```

### Step 5: Initialize Database Tables

```bash
python database/init_db.py
```

This will:
1. ✅ Test database connection
2. ✅ Create all required tables
3. ✅ Show database information

**Expected Output:**
```
============================================================
PORT CONGESTION DATABASE INITIALIZATION
============================================================

[1/3] Testing database connection...
Database connection successful!

[2/3] Database Information:
   Host: localhost
   Port: 5432
   Database: port_congestion_db
   User: postgres
   Connected: True

[3/3] Initializing database tables...
Creating database tables...
Database tables created successfully!

✅ Database initialized successfully!
```

### Step 6: Verify Setup

Test the connection:

```python
from database.db_connection import test_connection, get_database_info

if test_connection():
    print("✅ Database connected!")
    info = get_database_info()
    print(f"Tables: {info['tables']}")
```

## 🗄️ Database Schema

### Tables Created

1. **`predictions`** - ML predictions
   - `id`, `timestamp`, `predicted_risk_class`, probabilities
   - `yard_utilization_ratio`, `avg_truck_wait_min`
   - `model_type`, `model_version`

2. **`optimization_recommendations`** - Resource recommendations
   - `id`, `window_id`, `start_time`, `end_time`
   - `current_yard_util`, `recommended_yard_util`
   - `recommended_cranes`, `expected_risk_reduction`
   - `optimization_run_id`, `implemented`

3. **`operational_decisions`** - Actual decisions
   - `id`, `recommendation_id`, `decision_timestamp`
   - `decision_type`, `actual_cranes_allocated`
   - `approval_status`, `outcome_notes`

4. **`optimization_runs`** - Run metadata
   - `id`, `run_id`, `run_timestamp`, `status`
   - Configuration parameters
   - Impact metrics (before/after)

## 🚀 Usage

### Automatic Saving

When you run optimization:
```bash
python scripts/optimize_resources.py
```

Results are **automatically saved** to:
- ✅ Files (CSV, JSON, TXT) in `output/optimization/`
- ✅ **PostgreSQL database** (if connected)

### Manual Database Operations

#### Save a Prediction
```python
from database.db_connection import get_db_session
from database.db_operations import save_prediction
from datetime import datetime

with get_db_session() as session:
    prediction = save_prediction(
        session=session,
        timestamp=datetime.utcnow(),
        time_window_start=datetime(2023, 1, 1, 0, 0),
        predicted_risk_class=1,  # 0=Low, 1=Medium, 2=High
        probability_low=0.2,
        probability_medium=0.6,
        probability_high=0.2,
        model_type="RandomForest",
        yard_utilization_ratio=0.85,
        avg_truck_wait_min=45.5
    )
```

#### Query Predictions
```python
from database.db_connection import get_db_session
from database.db_operations import get_latest_predictions

with get_db_session() as session:
    predictions = get_latest_predictions(session, limit=100)
    for pred in predictions:
        print(f"{pred.timestamp}: {pred.predicted_risk_label}")
```

#### Query Optimization Results
```python
from database.db_connection import get_db_session
from database.db_operations import (
    get_latest_optimization_run,
    get_optimization_recommendations_by_run
)

with get_db_session() as session:
    latest_run = get_latest_optimization_run(session)
    if latest_run:
        recommendations = get_optimization_recommendations_by_run(
            session, latest_run.run_id
        )
        print(f"Found {len(recommendations)} recommendations")
```

## 🔍 Viewing Data in pgAdmin

1. Open **pgAdmin**
2. Connect to server
3. Expand **Databases** → **port_congestion_db** → **Schemas** → **public** → **Tables**
4. Right-click table → **View/Edit Data** → **All Rows**

## 🛠️ Troubleshooting

### Error: "Connection refused"
**Solution:**
- Check PostgreSQL is running: `pg_isready` or check Services
- Verify host/port: Default is `localhost:5432`
- Check firewall settings

### Error: "Authentication failed"
**Solution:**
- Verify username and password
- Check `pg_hba.conf` (usually in PostgreSQL data directory)
- Try: `psql -U postgres -d port_congestion_db`

### Error: "Database does not exist"
**Solution:**
```sql
CREATE DATABASE port_congestion_db;
```

### Error: "Module 'psycopg2' not found"
**Solution:**
```bash
pip install psycopg2-binary
```

### Error: "Permission denied"
**Solution:**
```sql
-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE port_congestion_db TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
```

## 📊 SQL Queries Examples

### Get Latest Predictions
```sql
SELECT * FROM predictions 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Get High-Risk Predictions
```sql
SELECT * FROM predictions 
WHERE predicted_risk_class = 2 
ORDER BY timestamp DESC;
```

### Get Unimplemented Recommendations
```sql
SELECT * FROM optimization_recommendations 
WHERE implemented = false 
ORDER BY start_time;
```

### Get Optimization Run Summary
```sql
SELECT 
    run_id,
    run_timestamp,
    status,
    total_recommendations,
    delay_risk_reduction_pct
FROM optimization_runs
ORDER BY run_timestamp DESC;
```

## ✅ Verification Checklist

- [ ] PostgreSQL installed and running
- [ ] Database `port_congestion_db` created
- [ ] Python packages installed (`psycopg2-binary`, `sqlalchemy`)
- [ ] Database credentials configured
- [ ] Connection test successful
- [ ] Tables initialized
- [ ] Optimization script saves to database
- [ ] Can query data from database

## 🎯 Next Steps

1. **Run optimization** to populate database
2. **Update dashboard** to read from database (optional)
3. **Create analytics queries** for reporting
4. **Set up backups** for production

## 📝 Quick Reference

**Connection String Format:**
```
postgresql://username:password@host:port/database
```

**Default Values:**
- Host: `localhost`
- Port: `5432`
- Database: `port_congestion_db`
- User: `postgres`
- Password: (set during installation)

**Key Files:**
- `database/models.py` - Database schema/models
- `database/db_connection.py` - Connection utilities
- `database/db_operations.py` - CRUD operations
- `database/init_db.py` - Initialization script

