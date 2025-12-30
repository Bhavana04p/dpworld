# FastAPI Backend Setup Guide

## ✅ What Was Created

Complete FastAPI backend structure:

```
api/
├── __init__.py
├── main.py                    # FastAPI app with all endpoints
├── schemas.py                 # Pydantic request/response models
├── database.py                # Database integration
├── requirements.txt           # Dependencies
└── services/
    ├── __init__.py
    ├── prediction_service.py  # ML prediction logic
    └── optimization_service.py  # Optimization logic
```

## 🚀 Setup Steps

### Step 1: Install Dependencies

```bash
pip install -r api/requirements.txt
```

Or install individually:
```bash
pip install fastapi uvicorn[standard] pydantic python-multipart
pip install sqlalchemy psycopg2-binary
pip install pandas numpy scikit-learn joblib
```

**Optional (for optimization):**
```bash
pip install ortools  # OR
pip install pulp
```

### Step 2: Verify Database Connection

Make sure PostgreSQL is running and database is accessible:
```bash
python database/test_db.py
```

### Step 3: Start the API Server

```bash
uvicorn api.main:app --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 4: Test the API

#### Option 1: Interactive Docs (Recommended)
1. Open browser: http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Fill in request body
4. Click "Execute"

#### Option 2: Command Line
```bash
# Health check
curl http://localhost:8000/health

# Predict delay risk
curl -X POST "http://localhost:8000/predict-delay-risk" \
  -H "Content-Type: application/json" \
  -d '{"yard_utilization_ratio": 0.85, "avg_truck_wait_min": 45.5}'
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/health`

Returns API status and database connection status.

### 2. Predict Delay Risk
**POST** `/predict-delay-risk`

Predicts delay risk (0=Low, 1=Medium, 2=High) based on operational features.

**Required fields:**
- `yard_utilization_ratio` (0-1)
- `avg_truck_wait_min` (>=0)

**Optional fields:**
- Weather: `wind_speed`, `rainfall`, `wave_height`, `temperature`, `visibility`
- Temporal: `hour_of_day`, `day_of_week`, `is_weekend`, `month`
- Crane/Berth: `available_cranes`, `berth_utilization`

**Query parameter:**
- `save_to_db` (bool): Save prediction to database

### 3. Optimize Resources
**POST** `/optimize-resources`

Provides optimization recommendations for resource allocation.

**Required fields:**
- `current_yard_utilization` (0-1)
- `available_cranes` (>=1)

**Optional fields:**
- `time_window_hours` (default: 24)
- `current_delay_risk`
- `avg_truck_wait_min`
- Configuration overrides

**Query parameter:**
- `save_to_db` (bool): Save optimization results to database

### 4. Model Info
**GET** `/models/info`

Returns information about loaded ML models.

## 🔧 Configuration

### Database

The API uses the same database configuration:
- Reads from `database/db_connection.py`
- Database: `port_congestion_db`
- Host: `localhost:5432`
- User: `postgres`
- Password: Set in `database/db_connection.py` or environment variable

### Environment Variables (Optional)

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=port_congestion_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

## 🧪 Testing

### Quick Test Script

```bash
python test_api.py
```

This will verify:
- All imports work
- Database connection
- Model loading

### Manual Testing

1. **Start server:**
   ```bash
   uvicorn api.main:app --reload
   ```

2. **Open interactive docs:**
   - Visit http://localhost:8000/docs
   - Test endpoints directly in browser

3. **Test with curl:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Predict
   curl -X POST "http://localhost:8000/predict-delay-risk?save_to_db=true" \
     -H "Content-Type: application/json" \
     -d '{"yard_utilization_ratio": 0.85, "avg_truck_wait_min": 45.5}'
   ```

## 📊 Integration

### With Existing System

- ✅ **Reuses ML Models**: Loads from `output/models/`
- ✅ **Reuses Optimization**: Uses `scripts/optimize_resources.py` logic
- ✅ **Database Integration**: Uses existing `database/` module
- ✅ **Works with Streamlit**: Does not replace dashboard

### Database Saving

When `save_to_db=true`:
- **Predictions**: Saved to `predictions` table
- **Optimization**: Saved to `optimization_runs` and `optimization_recommendations` tables

## 🚀 Production Deployment

### For AWS/Azure:

1. **Install dependencies:**
   ```bash
   pip install -r api/requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export DATABASE_URL=postgresql://user:pass@host:port/db
   ```

3. **Run with production server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

4. **Use process manager** (systemd, supervisor, etc.)
5. **Set up reverse proxy** (nginx)
6. **Enable HTTPS**

## ✅ Verification Checklist

- [ ] Dependencies installed (`pip install -r api/requirements.txt`)
- [ ] Database connection works
- [ ] API server starts (`uvicorn api.main:app --reload`)
- [ ] Health check returns "ok"
- [ ] Interactive docs accessible at `/docs`
- [ ] Prediction endpoint works
- [ ] Optimization endpoint works
- [ ] Database saving works (when `save_to_db=true`)

## 🎯 Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r api/requirements.txt
   ```

2. **Start the server:**
   ```bash
   uvicorn api.main:app --reload
   ```

3. **Test endpoints:**
   - Visit http://localhost:8000/docs
   - Try all endpoints

4. **Integrate with frontend:**
   - Use API endpoints in web applications
   - Connect dashboards to API

5. **Deploy to cloud:**
   - AWS/Azure deployment
   - Set up monitoring
   - Add authentication if needed

## 📝 Notes

- API is stateless and can be scaled horizontally
- Model is loaded once and cached
- Database connections use connection pooling
- All endpoints have proper validation and error handling
- Interactive docs available at `/docs`

## 🆘 Troubleshooting

### "No module named 'fastapi'"
**Solution:** Install dependencies:
```bash
pip install -r api/requirements.txt
```

### "Model not found"
**Solution:** Ensure ML model exists in `output/models/`

### "Database connection failed"
**Solution:** 
- Check PostgreSQL is running
- Verify credentials in `database/db_connection.py`
- Test connection: `python database/test_db.py`

### "Optimization failed"
**Solution:**
- Install optimization solver: `pip install ortools` or `pip install pulp`
- Or use heuristic mode (no solver needed)

## 🎉 You're Ready!

Your FastAPI backend is complete and ready to use. Just install dependencies and start the server!

