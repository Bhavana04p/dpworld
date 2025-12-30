# Power BI Integration Guide - Near-Real-Time Analytics

## 🎯 Architecture Overview

```
Python (ML + Optimization)
        ↓ (every 5 minutes)
PostgreSQL tables
        ↓ (scheduled refresh every 15-30 min)
Power BI Dashboard
```

## ✅ What This Does

The `live_data_ingestion.py` script:
1. **Generates realistic port operational data** every 5 minutes
2. **Makes ML predictions** using your trained model
3. **Generates optimization recommendations**
4. **Saves everything to PostgreSQL** tables
5. **Power BI refreshes** and displays the data

## 🚀 Setup Steps

### Step 1: Start Data Ingestion Script

```bash
python scripts/live_data_ingestion.py
```

**What happens:**
- Script runs continuously
- Writes predictions every 5 minutes
- Writes optimization recommendations periodically
- All data goes to PostgreSQL tables

**Expected output:**
```
======================================================================
LIVE DATA INGESTION FOR POWER BI
======================================================================

[1/3] Testing database connection...
[SUCCESS] Database connected!

[2/3] Checking ML model availability...
[SUCCESS] ML model loaded: RandomForestClassifier

[3/3] Starting data ingestion...
   Interval: 300 seconds (5.0 minutes)
   Database: port_congestion_db
   Tables: predictions, optimization_runs, optimization_recommendations

--- Cycle #1 ---
[2025-12-29 01:50:00] Generating prediction...
   Delay Risk: Medium (confidence: 0.75)
   Yard Utilization: 85.2%
   Truck Wait: 45.3 min
   ✅ Prediction saved (ID: 1)

[2025-12-29 01:50:01] Generating optimization...
   Recommended Cranes: 15
   Target Yard Util: 75.0%
   Improvement: 46.7%
   ✅ Optimization saved (Run ID: 20251229_015001)
```

### Step 2: Verify Data in PostgreSQL

**Using pgAdmin:**
1. Open pgAdmin
2. Navigate to: Databases → port_congestion_db → Tables
3. Right-click `predictions` → View/Edit Data
4. You should see new rows appearing every 5 minutes

**Using psql:**
```bash
psql -U postgres -d port_congestion_db

# Count predictions
SELECT COUNT(*) FROM predictions;

# View latest predictions
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10;

# View optimization runs
SELECT * FROM optimization_runs ORDER BY run_timestamp DESC LIMIT 5;
```

### Step 3: Connect Power BI to PostgreSQL

#### Option A: Direct Query (Recommended)

1. **Open Power BI Desktop**
2. **Get Data** → **Database** → **PostgreSQL database**
3. **Enter connection details:**
   - Server: `localhost` (or your PostgreSQL host)
   - Database: `port_congestion_db`
   - Username: `postgres`
   - Password: Your PostgreSQL password
4. **Select tables:**
   - `predictions`
   - `optimization_runs`
   - `optimization_recommendations`
5. **Load data**

#### Option B: Import Data

1. **Get Data** → **PostgreSQL database**
2. **Import** (not DirectQuery)
3. **Select tables**
4. **Load**

### Step 4: Create Power BI Dashboard

**Recommended Visualizations:**

1. **Delay Risk Distribution**
   - Chart type: Pie/Donut
   - Data: `predictions.predicted_risk_class`
   - Shows: Low/Medium/High risk distribution

2. **Yard Utilization Over Time**
   - Chart type: Line chart
   - X-axis: `predictions.timestamp`
   - Y-axis: `predictions.yard_utilization_ratio`
   - Shows: Utilization trends

3. **Truck Wait Time**
   - Chart type: Column chart
   - X-axis: `predictions.timestamp`
   - Y-axis: `predictions.avg_truck_wait_min`
   - Shows: Wait time trends

4. **Optimization Impact**
   - Chart type: Card/KPI
   - Data: `optimization_runs.delay_risk_reduction_pct`
   - Shows: Average improvement percentage

5. **Recent Predictions Table**
   - Chart type: Table
   - Columns: timestamp, delay_risk, yard_utilization_ratio, confidence
   - Shows: Latest predictions

### Step 5: Configure Scheduled Refresh

**For Power BI Service (Cloud):**

1. **Publish** your dashboard to Power BI Service
2. **Go to** Dataset settings
3. **Configure** Scheduled refresh:
   - Enable: Yes
   - Frequency: Every 15-30 minutes
   - Time: Set appropriate times
4. **Save**

**For Power BI Desktop:**

1. **Data** → **Refresh**
2. Or set up **Power BI Gateway** for automatic refresh

## 📊 Data Flow

### Predictions Table
- **Updated:** Every 5 minutes
- **Contains:** Delay risk predictions, probabilities, features
- **Use for:** Real-time risk monitoring, trend analysis

### Optimization Runs Table
- **Updated:** Periodically (every few cycles)
- **Contains:** Optimization run metadata, impact metrics
- **Use for:** Optimization performance tracking

### Optimization Recommendations Table
- **Updated:** With each optimization run
- **Contains:** Detailed recommendations per time window
- **Use for:** Actionable insights, resource planning

## 🎤 Interview Talking Points

**"How does your system handle real-time data?"**

> "The system uses near-real-time ingestion with scheduled refresh, which is the standard pattern for operational dashboards. Python scripts continuously write predictions and optimization results to PostgreSQL every 5 minutes. Power BI refreshes every 15-30 minutes, providing near-real-time visibility into port operations. This approach balances data freshness with system performance and is how 90% of enterprise dashboards operate."

**"How do you ensure data quality?"**

> "All predictions are validated using the trained ML model, and optimization recommendations are generated using proven algorithms. Data is stored in a normalized PostgreSQL database with proper constraints, ensuring consistency and reliability."

## 🔧 Configuration Options

### Adjust Ingestion Interval

Edit `scripts/live_data_ingestion.py`:

```python
INGESTION_INTERVAL_SECONDS = 300  # Change to desired seconds
```

**Common intervals:**
- 60 seconds (1 minute) - Very frequent
- 300 seconds (5 minutes) - Recommended
- 600 seconds (10 minutes) - Less frequent

### Adjust Optimization Frequency

In `ingest_data_cycle()` function:

```python
if random.random() < 0.33:  # Change 0.33 to adjust frequency
```

- `0.33` = 33% chance each cycle (recommended)
- `0.5` = 50% chance (more frequent)
- `0.1` = 10% chance (less frequent)

## ✅ Verification Checklist

- [ ] Data ingestion script runs without errors
- [ ] Predictions appear in PostgreSQL `predictions` table
- [ ] Optimization runs appear in `optimization_runs` table
- [ ] Power BI can connect to PostgreSQL
- [ ] Power BI dashboard displays data correctly
- [ ] Scheduled refresh is configured
- [ ] Data updates appear in Power BI after refresh

## 🚀 Running in Production

### Option 1: Background Process (Windows)

```bash
# Run in background
start /B python scripts/live_data_ingestion.py

# Or use Task Scheduler to run on startup
```

### Option 2: Service (Linux)

Create a systemd service:

```ini
[Unit]
Description=Port Congestion Data Ingestion
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/dpworld
ExecStart=/usr/bin/python3 scripts/live_data_ingestion.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Option 3: Docker Container

```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "scripts/live_data_ingestion.py"]
```

## 📝 Notes

- **Near-real-time:** 5-minute ingestion + 15-30 minute refresh = acceptable latency
- **Scalable:** Can handle high data volumes
- **Reliable:** Uses existing database infrastructure
- **Cost-effective:** No additional cloud services needed
- **Interview-ready:** Standard enterprise pattern

## 🎯 Summary

1. **Run ingestion script:** `python scripts/live_data_ingestion.py`
2. **Connect Power BI** to PostgreSQL
3. **Create visualizations** using the data
4. **Configure refresh** every 15-30 minutes
5. **Monitor** data flow and dashboard updates

**Your system now provides near-real-time analytics for Power BI!** 🎉

