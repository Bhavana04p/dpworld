# FastAPI Backend - Port Congestion Optimization System

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Start the API Server

```bash
uvicorn api.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predict Delay Risk
```bash
curl -X POST "http://localhost:8000/predict-delay-risk" \
  -H "Content-Type: application/json" \
  -d '{
    "yard_utilization_ratio": 0.85,
    "avg_truck_wait_min": 45.5,
    "wind_speed": 15.0,
    "hour_of_day": 14
  }'
```

#### Optimize Resources
```bash
curl -X POST "http://localhost:8000/optimize-resources" \
  -H "Content-Type: application/json" \
  -d '{
    "current_yard_utilization": 0.85,
    "available_cranes": 20,
    "current_delay_risk": 1.5,
    "avg_truck_wait_min": 45.0
  }'
```

## 📋 API Endpoints

### 1. Health Check
**GET** `/health`

Returns API and database status.

**Response:**
```json
{
  "status": "ok",
  "database": "connected"
}
```

### 2. Predict Delay Risk
**POST** `/predict-delay-risk`

Predicts 24-hour delay risk based on operational features.

**Request Body:**
```json
{
  "yard_utilization_ratio": 0.85,
  "avg_truck_wait_min": 45.5,
  "wind_speed": 15.0,
  "hour_of_day": 14,
  "day_of_week": 2
}
```

**Response:**
```json
{
  "delay_risk": 1,
  "confidence": 0.75,
  "probabilities": {
    "low": 0.15,
    "medium": 0.75,
    "high": 0.10
  },
  "prediction_id": 123
}
```

**Query Parameters:**
- `save_to_db` (bool, default: false): Save prediction to database

### 3. Optimize Resources
**POST** `/optimize-resources`

Provides optimization recommendations for resource allocation.

**Request Body:**
```json
{
  "time_window_hours": 24,
  "current_yard_utilization": 0.85,
  "available_cranes": 20,
  "current_delay_risk": 1.5,
  "avg_truck_wait_min": 45.0
}
```

**Response:**
```json
{
  "status": "optimal",
  "solver_used": "OR-Tools",
  "recommended_cranes": 15,
  "yard_utilization_target": 0.75,
  "delay_risk_before": 1.5,
  "delay_risk_after": 0.8,
  "improvement_percent": 46.7,
  "recommendations": [...],
  "optimization_run_id": "20251229_001050"
}
```

**Query Parameters:**
- `save_to_db` (bool, default: false): Save optimization results to database

### 4. Model Info
**GET** `/models/info`

Returns information about loaded ML models.

## 🔧 Configuration

### Database Connection

The API uses the same database configuration as the main application:
- Reads from `database/db_connection.py`
- Uses environment variables or default values
- Database: `port_congestion_db`
- Host: `localhost:5432`

### Environment Variables

Set these if needed:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=port_congestion_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

## 🏗️ Architecture

```
api/
├── main.py                 # FastAPI application and endpoints
├── schemas.py              # Pydantic request/response models
├── database.py             # Database connection utilities
├── services/
│   ├── prediction_service.py    # ML prediction logic
│   └── optimization_service.py  # Optimization logic
└── requirements.txt        # Dependencies
```

## 🔄 Integration with Existing System

- **Reuses ML Models**: Loads models from `output/models/`
- **Reuses Optimization**: Uses logic from `scripts/optimize_resources.py`
- **Database Integration**: Uses existing `database/` module
- **Works Alongside Streamlit**: Does not replace the dashboard

## 📊 Database Integration

### Saving Predictions

When `save_to_db=true`:
- Predictions saved to `predictions` table
- Includes all feature values and probabilities

### Saving Optimization Results

When `save_to_db=true`:
- Optimization runs saved to `optimization_runs` table
- Recommendations saved to `optimization_recommendations` table
- Links recommendations to optimization runs

## 🧪 Testing

### Using Interactive Docs

1. Start the server: `uvicorn api.main:app --reload`
2. Open http://localhost:8000/docs
3. Click "Try it out" on any endpoint
4. Fill in request body
5. Click "Execute"

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict delay risk
response = requests.post(
    "http://localhost:8000/predict-delay-risk",
    json={
        "yard_utilization_ratio": 0.85,
        "avg_truck_wait_min": 45.5
    }
)
print(response.json())
```

## 🚀 Production Deployment

### For AWS/Azure Deployment:

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
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

4. **Use process manager (e.g., systemd, supervisor)**
5. **Set up reverse proxy (nginx)**
6. **Enable HTTPS**

## 📝 Notes

- The API is stateless and can be scaled horizontally
- Model is loaded once and cached in memory
- Database connections are managed via connection pooling
- All endpoints return proper HTTP status codes
- Input validation is handled by Pydantic schemas

## ✅ Verification Checklist

- [ ] API server starts without errors
- [ ] Health check returns "ok"
- [ ] Database connection works
- [ ] Prediction endpoint returns valid results
- [ ] Optimization endpoint returns valid results
- [ ] Interactive docs accessible at `/docs`
- [ ] Predictions can be saved to database
- [ ] Optimization results can be saved to database

## 🎯 Next Steps

1. **Test all endpoints** using interactive docs
2. **Integrate with frontend** applications
3. **Add authentication** if needed
4. **Deploy to cloud** (AWS/Azure)
5. **Set up monitoring** and logging
6. **Add rate limiting** for production

