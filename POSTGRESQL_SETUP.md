# PostgreSQL Database Setup - Complete Guide

## 🎯 Overview

Your system now uses **PostgreSQL** to store:
- ✅ ML predictions (delay risk forecasts)
- ✅ Optimization recommendations (crane allocation, yard utilization)
- ✅ Operational decisions (actual actions taken)
- ✅ Optimization run history

## 📋 Step-by-Step Setup

### Step 1: Install PostgreSQL

#### Windows
1. Download: https://www.postgresql.org/download/windows/
2. Run installer
3. **Remember the password** you set for `postgres` user
4. Default port: **5432**

#### Verify Installation
```bash
psql --version
```

### Step 2: Create Database

Open **Command Prompt** or **PowerShell** and run:

```bash
psql -U postgres
```

Enter your PostgreSQL password, then:

```sql
CREATE DATABASE port_congestion_db;
\q
```

**Or using pgAdmin (GUI):**
1. Open pgAdmin
2. Right-click "Databases" → "Create" → "Database"
3. Name: `port_congestion_db`
4. Click "Save"

### Step 3: Install Python Packages

```bash
pip install psycopg2-binary sqlalchemy
```

Or update requirements:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Database Connection

#### Option A: Set Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:DB_PASSWORD="your_postgres_password_here"
```

**Windows CMD:**
```cmd
set DB_PASSWORD=your_postgres_password_here
```

#### Option B: Edit Configuration File

Open `database/db_connection.py` and change line 15:

```python
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_postgres_password_here')
```

Replace `'your_postgres_password_here'` with your actual PostgreSQL password.

### Step 5: Initialize Database Tables

```bash
python database/init_db.py
```

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

### Step 6: Test Database Connection

```bash
python database/test_db.py
```

This will test:
- ✅ Connection
- ✅ Saving predictions
- ✅ Saving optimization runs
- ✅ Querying data

## 🚀 Using the Database

### Automatic Saving

When you run optimization:
```bash
python scripts/optimize_resources.py
```

**Results are automatically saved to:**
1. ✅ Files: `output/optimization/*.csv`, `*.json`
2. ✅ **PostgreSQL Database**: All tables

### Dashboard Integration

The Streamlit dashboard **automatically**:
- ✅ Reads from PostgreSQL if connected
- ✅ Falls back to files if database unavailable
- ✅ Shows database status in sidebar

**No code changes needed!** Just run:
```bash
streamlit run streamlit_app/app.py
```

## 📊 Database Tables

### 1. `optimization_runs`
Stores metadata about each optimization execution:
- Run ID, timestamp, status
- Configuration parameters
- Impact metrics (before/after)

### 2. `optimization_recommendations`
Stores all recommendations:
- Time windows
- Current vs recommended yard utilization
- Recommended crane allocation
- Expected risk reduction

### 3. `predictions`
Stores ML model predictions (for future use)

### 4. `operational_decisions`
Stores actual decisions made (for future use)

## 🔍 Viewing Data

### Using pgAdmin (GUI)

1. Open **pgAdmin**
2. Connect to PostgreSQL server
3. Navigate: **Databases** → **port_congestion_db** → **Schemas** → **public** → **Tables**
4. Right-click any table → **View/Edit Data** → **All Rows**

### Using psql (Command Line)

```bash
psql -U postgres -d port_congestion_db
```

```sql
-- View all optimization runs
SELECT * FROM optimization_runs ORDER BY run_timestamp DESC LIMIT 5;

-- View recommendations from latest run
SELECT * FROM optimization_recommendations 
ORDER BY start_time DESC LIMIT 10;

-- Count total recommendations
SELECT COUNT(*) FROM optimization_recommendations;

-- Exit
\q
```

### Using Python

```python
from database.db_connection import get_db_session
from database.db_operations import get_latest_optimization_run

with get_db_session() as session:
    run = get_latest_optimization_run(session)
    print(f"Latest run: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Recommendations: {run.total_recommendations}")
```

## 🛠️ Troubleshooting

### "Connection refused"
- ✅ Check PostgreSQL is running (Windows: Services → postgresql)
- ✅ Verify host: `localhost`, port: `5432`

### "Authentication failed"
- ✅ Check username: `postgres`
- ✅ Verify password is correct
- ✅ Try: `psql -U postgres` to test login

### "Database does not exist"
```sql
CREATE DATABASE port_congestion_db;
```

### "Module 'psycopg2' not found"
```bash
pip install psycopg2-binary
```

### "Permission denied"
```sql
GRANT ALL PRIVILEGES ON DATABASE port_congestion_db TO postgres;
```

## ✅ Verification Checklist

- [ ] PostgreSQL installed and running
- [ ] Database `port_congestion_db` created
- [ ] Python packages installed (`psycopg2-binary`, `sqlalchemy`)
- [ ] Database password configured
- [ ] `python database/init_db.py` completed successfully
- [ ] `python database/test_db.py` passed all tests
- [ ] Optimization script saves to database
- [ ] Dashboard shows "🗄️ Database: Connected" in sidebar

## 📝 Quick Reference

**Default Connection Settings:**
- Host: `localhost`
- Port: `5432`
- Database: `port_congestion_db`
- User: `postgres`
- Password: (set during PostgreSQL installation)

**Key Commands:**
```bash
# Initialize database
python database/init_db.py

# Test connection
python database/test_db.py

# Run optimization (saves to DB)
python scripts/optimize_resources.py

# Start dashboard (reads from DB)
streamlit run streamlit_app/app.py
```

## 🎯 What Happens Now

1. **Optimization runs** → Automatically saved to PostgreSQL
2. **Dashboard** → Automatically reads from PostgreSQL
3. **All data** → Stored in database tables
4. **Files** → Still created as backup

**You're all set! 🎉**

