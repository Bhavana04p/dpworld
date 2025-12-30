# ✅ Next Steps After Database Connection

## 🎯 What You Can Do Now

### 1. Run Optimization (Saves to Database)

Now that your database is connected, when you run the optimization script, **all results will automatically be saved to PostgreSQL**:

```bash
python scripts/optimize_resources.py
```

**What gets saved:**
- ✅ All optimization recommendations (crane allocation, yard utilization)
- ✅ Optimization run metadata (timestamp, status, configuration)
- ✅ Impact metrics (before/after comparisons)
- ✅ Results are also saved to files as backup

### 2. View Data in Dashboard

The dashboard **automatically reads from the database**:

```bash
streamlit run streamlit_app/app.py
```

**Navigate to "🎯 Optimization & Recommendations" page:**
- Data loads from PostgreSQL automatically
- Shows all recommendations from the database
- Displays optimization history

### 3. View Data in PostgreSQL

#### Option A: Using pgAdmin (GUI - Easiest)

1. Open **pgAdmin**
2. Connect to PostgreSQL server
3. Navigate: **Databases** → **port_congestion_db** → **Schemas** → **public** → **Tables**
4. Right-click `optimization_recommendations` → **View/Edit Data** → **All Rows**

#### Option B: Using psql (Command Line)

```bash
psql -U postgres -d port_congestion_db
```

Then run SQL queries:
```sql
-- View all optimization runs
SELECT * FROM optimization_runs ORDER BY run_timestamp DESC LIMIT 5;

-- View recommendations
SELECT * FROM optimization_recommendations LIMIT 10;

-- Count total records
SELECT COUNT(*) FROM optimization_recommendations;

-- Exit
\q
```

### 4. Query Data with Python

```python
from database.db_connection import get_db_session
from database.db_operations import (
    get_latest_optimization_run,
    get_optimization_recommendations_by_run
)

with get_db_session() as session:
    # Get latest optimization run
    latest_run = get_latest_optimization_run(session)
    if latest_run:
        print(f"Latest Run: {latest_run.run_id}")
        print(f"Status: {latest_run.status}")
        print(f"Recommendations: {latest_run.total_recommendations}")
        
        # Get all recommendations for this run
        recommendations = get_optimization_recommendations_by_run(
            session, latest_run.run_id
        )
        print(f"Found {len(recommendations)} recommendations")
```

## 📊 Complete Workflow

### Step 1: Run Optimization
```bash
python scripts/optimize_resources.py
```

**Expected Output:**
```
[INFO] Loading ML dataset...
[INFO] Preparing optimization data...
[INFO] Running optimization...
[SUCCESS] Optimization results saved:
   - Recommendations: output/optimization/recommendations_*.csv
   - Impact Analysis: output/optimization/impact_analysis_*.json
   - Summary: output/optimization/summary_*.txt
   - Database: Saved to PostgreSQL (run_id: 20251229_001050)
```

### Step 2: View in Dashboard
```bash
streamlit run streamlit_app/app.py
```

1. Check sidebar - Should show "✅ Connected"
2. Go to "🎯 Optimization & Recommendations" page
3. See all recommendations loaded from database

### Step 3: Verify in Database

**Using pgAdmin:**
- Open pgAdmin → Connect → Navigate to tables
- View `optimization_recommendations` table
- See all your optimization results

## 🔍 What Data is Stored

### Table: `optimization_runs`
- Run ID (unique identifier)
- Timestamp
- Status (optimal/heuristic/infeasible)
- Configuration used
- Impact metrics

### Table: `optimization_recommendations`
- Time window details
- Current vs recommended yard utilization
- Recommended crane allocation
- Expected risk reduction
- Links to optimization run

### Table: `predictions`
- Ready for storing ML predictions (future use)

### Table: `operational_decisions`
- Ready for storing actual decisions made (future use)

## 🎯 Recommended Actions

1. **Run your first optimization:**
   ```bash
   python scripts/optimize_resources.py
   ```

2. **Check the dashboard:**
   - Open Streamlit dashboard
   - Go to Optimization page
   - Verify data appears from database

3. **Verify in PostgreSQL:**
   - Use pgAdmin or psql to view the data
   - Confirm records are saved

4. **Run multiple optimizations:**
   - Each run creates a new record
   - Compare different optimization runs
   - Track optimization history

## 💡 Tips

- **Backup regularly**: Use `pg_dump` to backup your database
- **Monitor size**: Check table sizes in pgAdmin
- **Query history**: All optimization runs are stored with timestamps
- **Compare runs**: Query different runs to compare results

## ✅ Verification Checklist

- [ ] Database shows "✅ Connected" in dashboard sidebar
- [ ] Ran optimization script successfully
- [ ] Optimization results saved to database
- [ ] Can view data in dashboard
- [ ] Can query data in PostgreSQL
- [ ] Multiple optimization runs can be stored

## 🚀 You're All Set!

Your system is now fully integrated with PostgreSQL. Every time you run optimization, results are automatically saved to the database and can be viewed in the dashboard or queried directly.

**Next time you run optimization, everything will be saved automatically!** 🎉

