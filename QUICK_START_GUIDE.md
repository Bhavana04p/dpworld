# 🚀 Quick Start Guide - After Database Connection

## ✅ Your Database is Connected!

Now you can:

## 1️⃣ Run Optimization (Saves to Database)

```bash
python scripts/optimize_resources.py
```

**What happens:**
- ✅ Optimization runs and finds optimal resource allocation
- ✅ Results saved to files: `output/optimization/*.csv`
- ✅ **Results automatically saved to PostgreSQL database**
- ✅ You'll see: `Database: Saved to PostgreSQL (run_id: ...)`

## 2️⃣ View Results in Dashboard

```bash
streamlit run streamlit_app/app.py
```

**Steps:**
1. Open the dashboard
2. Check sidebar - Should show "✅ Connected"
3. Navigate to **"🎯 Optimization & Recommendations"** page
4. See all recommendations loaded from database!

## 3️⃣ View Data in PostgreSQL

### Using pgAdmin (Easiest):
1. Open **pgAdmin**
2. Connect to server
3. Navigate: **Databases** → **port_congestion_db** → **Tables**
4. Right-click `optimization_recommendations` → **View/Edit Data**

### Using psql:
```bash
psql -U postgres -d port_congestion_db

# View latest recommendations
SELECT * FROM optimization_recommendations ORDER BY start_time DESC LIMIT 10;

# View optimization runs
SELECT * FROM optimization_runs ORDER BY run_timestamp DESC;
```

## 📊 What Gets Stored

Every time you run optimization:

1. **`optimization_runs` table:**
   - Run ID, timestamp, status
   - Configuration used
   - Impact metrics (before/after)

2. **`optimization_recommendations` table:**
   - All recommendations per time window
   - Current vs recommended values
   - Expected improvements

## 🎯 Complete Example Workflow

### Step 1: Run Optimization
```bash
python scripts/optimize_resources.py
```

**Output:**
```
[INFO] Loading ML dataset...
[INFO] Preparing optimization data...
[INFO] Running optimization with OR-Tools...
[SUCCESS] Optimization completed!
[SUCCESS] Optimization results saved:
   - Recommendations: output/optimization/recommendations_20251229_001050.csv
   - Impact Analysis: output/optimization/impact_analysis_20251229_001050.json
   - Summary: output/optimization/summary_20251229_001050.txt
   - Database: Saved to PostgreSQL (run_id: 20251229_001050)
```

### Step 2: View in Dashboard
1. Open dashboard: `streamlit run streamlit_app/app.py`
2. Go to "🎯 Optimization & Recommendations"
3. See all data loaded from database!

### Step 3: Verify in Database
```sql
-- In psql or pgAdmin
SELECT COUNT(*) FROM optimization_recommendations;
-- Should show number of recommendations

SELECT run_id, run_timestamp, status, total_recommendations 
FROM optimization_runs 
ORDER BY run_timestamp DESC;
-- Shows all optimization runs
```

## 🔄 Run Multiple Times

You can run optimization multiple times:
- Each run creates a new record in `optimization_runs`
- All recommendations are stored with a unique `run_id`
- Compare different optimization runs
- Track optimization history over time

## ✅ Verification

After running optimization, verify:

1. **Check files:**
   ```bash
   ls output/optimization/
   ```
   Should see: `recommendations_*.csv`, `impact_analysis_*.json`, `summary_*.txt`

2. **Check database:**
   ```sql
   SELECT COUNT(*) FROM optimization_recommendations;
   ```
   Should show number of recommendations

3. **Check dashboard:**
   - Open dashboard
   - Go to Optimization page
   - Should see recommendations displayed

## 🎉 You're Ready!

**Everything is set up. Just run:**
```bash
python scripts/optimize_resources.py
```

**And your results will be automatically saved to PostgreSQL!** 🚀

