# PostgreSQL Database Setup - Everything You Need to Know

## 🎯 What You're Setting Up

PostgreSQL database integration that stores:
- ✅ **Optimization Recommendations** - All resource allocation recommendations
- ✅ **Optimization Runs** - Metadata about each optimization execution
- ✅ **Predictions** - ML model predictions (ready for future use)
- ✅ **Operational Decisions** - Actual decisions made (ready for future use)

## 📋 Complete Setup Steps

### STEP 1: Install PostgreSQL

#### Windows
1. **Download**: https://www.postgresql.org/download/windows/
   - Choose latest version (15 or 16)
   - Download the installer

2. **Install**:
   - Run the installer
   - Choose installation directory (default is fine)
   - **IMPORTANT**: Remember the password you set for `postgres` user!
   - Port: 5432 (default)
   - Locale: Default

3. **Verify Installation**:
   ```bash
   psql --version
   ```
   Should show: `psql (PostgreSQL) 15.x` or similar

#### Check if PostgreSQL is Running
- **Windows**: Open Services (`services.msc`) → Look for "postgresql-x64-15" → Should be "Running"
- Or check in pgAdmin (should open automatically after install)

---

### STEP 2: Create Database

#### Method 1: Using Command Line (psql)

1. **Open Command Prompt or PowerShell**

2. **Connect to PostgreSQL**:
   ```bash
   psql -U postgres
   ```
   Enter your PostgreSQL password when prompted

3. **Create Database**:
   ```sql
   CREATE DATABASE port_congestion_db;
   ```

4. **Verify**:
   ```sql
   \l
   ```
   You should see `port_congestion_db` in the list

5. **Exit**:
   ```sql
   \q
   ```

#### Method 2: Using pgAdmin (GUI - Easier)

1. **Open pgAdmin** (installed with PostgreSQL)

2. **Connect to Server**:
   - Right-click "Servers" → "Register" → "Server"
   - Name: `PostgreSQL 15` (or your version)
   - Host: `localhost`
   - Port: `5432`
   - Username: `postgres`
   - Password: (your PostgreSQL password)
   - Click "Save"

3. **Create Database**:
   - Right-click "Databases" → "Create" → "Database"
   - Database name: `port_congestion_db`
   - Click "Save"

---

### STEP 3: Install Python Packages

Open Command Prompt/PowerShell in your project directory:

```bash
pip install psycopg2-binary sqlalchemy
```

**Or install all requirements:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import psycopg2; import sqlalchemy; print('✅ Packages installed')"
```

---

### STEP 4: Configure Database Connection

You need to set your PostgreSQL password. Choose one method:

#### Method 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:DB_PASSWORD="your_postgres_password_here"
```

**Windows CMD:**
```cmd
set DB_PASSWORD=your_postgres_password_here
```

**Note**: This is temporary. For permanent setup, add to system environment variables.

#### Method 2: Edit Configuration File

1. Open `database/db_connection.py`

2. Find line 20:
   ```python
   DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
   ```

3. Change to your actual password:
   ```python
   DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_actual_password_here')
   ```

**⚠️ Security Note**: Don't commit passwords to git! Use environment variables for production.

---

### STEP 5: Initialize Database Tables

Run the initialization script:

```bash
python database/init_db.py
```

**What it does:**
1. Tests database connection
2. Creates 4 tables:
   - `predictions`
   - `optimization_recommendations`
   - `operational_decisions`
   - `optimization_runs`

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

**If you see errors:**
- Check PostgreSQL is running
- Verify password is correct
- Ensure database exists

---

### STEP 6: Test Database Connection

```bash
python database/test_db.py
```

**Expected Output:**
```
✅ Connection successful!
✅ Saved prediction ID: 1
✅ Retrieved 1 predictions
✅ Saved optimization run ID: test_run_001
✅ ALL TESTS PASSED!
```

---

### STEP 7: Run Optimization (Saves to Database)

```bash
python scripts/optimize_resources.py
```

**What happens:**
- ✅ Optimization runs
- ✅ Results saved to files: `output/optimization/*.csv`
- ✅ **Results saved to PostgreSQL database automatically**

**Look for this in output:**
```
[SUCCESS] Optimization results saved:
   - Recommendations: output/optimization/recommendations_*.csv
   - Impact Analysis: output/optimization/impact_analysis_*.json
   - Summary: output/optimization/summary_*.txt
   - Database: Saved to PostgreSQL (run_id: 20251229_001050)
```

---

### STEP 8: View Data in Dashboard

```bash
streamlit run streamlit_app/app.py
```

**In the sidebar, you'll see:**
- ✅ **"🗄️ Database: Connected"** (if database is working)
- ✅ **"🗄️ Database: Using files"** (if database not connected)

**Navigate to "🎯 Optimization & Recommendations"** page:
- Data loads from PostgreSQL automatically
- Falls back to files if database unavailable

---

## 🔍 Viewing Data in PostgreSQL

### Option 1: pgAdmin (GUI - Easiest)

1. Open **pgAdmin**
2. Connect to server
3. Navigate: **Databases** → **port_congestion_db** → **Schemas** → **public** → **Tables**
4. Right-click `optimization_recommendations` → **View/Edit Data** → **All Rows**

### Option 2: psql (Command Line)

```bash
psql -U postgres -d port_congestion_db
```

```sql
-- View all optimization runs
SELECT run_id, run_timestamp, status, total_recommendations 
FROM optimization_runs 
ORDER BY run_timestamp DESC 
LIMIT 5;

-- View recommendations
SELECT window_id, start_time, recommended_cranes, expected_risk_reduction
FROM optimization_recommendations
ORDER BY start_time DESC
LIMIT 10;

-- Count total records
SELECT COUNT(*) FROM optimization_recommendations;

-- Exit
\q
```

### Option 3: Python Script

```python
from database.db_connection import get_db_session
from database.db_operations import get_latest_optimization_run

with get_db_session() as session:
    run = get_latest_optimization_run(session)
    if run:
        print(f"Latest Run: {run.run_id}")
        print(f"Status: {run.status}")
        print(f"Recommendations: {run.total_recommendations}")
        print(f"Risk Reduction: {run.delay_risk_reduction_pct}%")
```

---

## 🛠️ Troubleshooting

### ❌ "Connection refused" or "Could not connect"
**Solutions:**
1. Check PostgreSQL is running:
   - Windows: Services → `postgresql-x64-15` → Should be "Running"
   - Or: `pg_isready` command
2. Verify host/port:
   - Default: `localhost:5432`
3. Check firewall settings

### ❌ "Authentication failed"
**Solutions:**
1. Verify username: `postgres`
2. Check password is correct
3. Test login: `psql -U postgres`
4. Check `pg_hba.conf` if needed

### ❌ "Database does not exist"
**Solution:**
```sql
CREATE DATABASE port_congestion_db;
```

### ❌ "Module 'psycopg2' not found"
**Solution:**
```bash
pip install psycopg2-binary
```

### ❌ "Permission denied"
**Solution:**
```sql
GRANT ALL PRIVILEGES ON DATABASE port_congestion_db TO postgres;
```

### ❌ "Table already exists"
**Solution:**
- This is normal if you've run init before
- Or drop and recreate: `python database/init_db.py` → Answer 'y' to drop existing

---

## 📊 Database Schema Overview

### Table: `optimization_runs`
Stores metadata about each optimization execution:
- `run_id` - Unique identifier (timestamp-based)
- `run_timestamp` - When optimization was run
- `status` - 'optimal', 'heuristic', 'infeasible'
- `solver_used` - 'OR-Tools', 'PuLP', 'Heuristic'
- Configuration parameters
- Impact metrics (before/after)

### Table: `optimization_recommendations`
Stores all recommendations:
- `window_id` - Time window identifier
- `start_time`, `end_time` - Time window
- `current_yard_util` - Current utilization
- `recommended_yard_util` - Target utilization
- `recommended_cranes` - Number of cranes
- `expected_risk_reduction` - Expected improvement
- `optimization_run_id` - Links to optimization run
- `implemented` - Whether recommendation was acted upon

### Table: `predictions`
Ready for storing ML predictions (future use)

### Table: `operational_decisions`
Ready for storing actual decisions (future use)

---

## ✅ Verification Checklist

Run through this checklist:

- [ ] PostgreSQL installed (`psql --version` works)
- [ ] PostgreSQL service is running
- [ ] Database `port_congestion_db` created
- [ ] Python packages installed (`psycopg2-binary`, `sqlalchemy`)
- [ ] Database password configured (environment variable or in code)
- [ ] `python database/init_db.py` completed successfully
- [ ] `python database/test_db.py` passed all tests
- [ ] `python scripts/optimize_resources.py` saves to database
- [ ] Dashboard shows "🗄️ Database: Connected"
- [ ] Can view data in pgAdmin or psql

---

## 🎯 Quick Commands Reference

```bash
# Initialize database
python database/init_db.py

# Test connection
python database/test_db.py

# Run optimization (saves to DB)
python scripts/optimize_resources.py

# Start dashboard (reads from DB)
streamlit run streamlit_app/app.py

# Connect to database (psql)
psql -U postgres -d port_congestion_db
```

---

## 📝 Connection Details

**Default Settings:**
- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `port_congestion_db`
- **User**: `postgres`
- **Password**: (set during PostgreSQL installation)

**Connection String Format:**
```
postgresql://postgres:password@localhost:5432/port_congestion_db
```

---

## 🎉 You're Done!

Once setup is complete:
1. ✅ Optimization automatically saves to PostgreSQL
2. ✅ Dashboard automatically reads from PostgreSQL
3. ✅ All data is stored in database tables
4. ✅ Files are still created as backup

**Your system now uses PostgreSQL for data storage! 🚀**

