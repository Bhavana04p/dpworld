# Quick Start: PostgreSQL Database Setup

## 🚀 5-Minute Setup

### 1. Install PostgreSQL
- **Windows**: Download from https://www.postgresql.org/download/windows/
- **Linux**: `sudo apt install postgresql` (Ubuntu/Debian)
- **macOS**: `brew install postgresql`

### 2. Create Database
```bash
psql -U postgres
CREATE DATABASE port_congestion_db;
\q
```

### 3. Install Python Packages
```bash
pip install psycopg2-binary sqlalchemy
```

### 4. Set Environment Variables

**Windows PowerShell:**
```powershell
$env:DB_PASSWORD="your_postgres_password"
```

**Linux/macOS:**
```bash
export DB_PASSWORD="your_postgres_password"
```

Or edit `database/db_connection.py` and set `DB_PASSWORD`.

### 5. Initialize Database
```bash
python database/init_db.py
```

### 6. Run Optimization (Saves to DB)
```bash
python scripts/optimize_resources.py
```

## ✅ Done!

Your optimization results are now stored in PostgreSQL!

## 🔍 Verify

```python
from database.db_connection import test_connection
test_connection()  # Should print "Database connection successful!"
```

## 📊 View in pgAdmin

1. Open pgAdmin
2. Connect to server
3. Navigate to: Databases → port_congestion_db → Schemas → public → Tables
4. Right-click `optimization_recommendations` → View/Edit Data

## 🎯 What Gets Stored

- ✅ All optimization recommendations
- ✅ Optimization run metadata
- ✅ Impact metrics (before/after)
- ✅ Configuration used

## 🔄 Dashboard Integration

The dashboard automatically reads from database if available, falls back to files if not.

