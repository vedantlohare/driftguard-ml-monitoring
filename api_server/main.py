from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import sys

# Add the stream_processor directory to path to import models
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stream_processor'))

from models import init_db, DataQualityAlert, DriftMetric

app = FastAPI(title="DriftGuard API")

# Allow CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URL = os.getenv('DB_URL', 'postgresql://admin:password@localhost:5433/driftguard')
SessionLocal = init_db(DB_URL)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/alerts/data_quality")
def get_data_quality_alerts(limit: int = 100, db: Session = Depends(get_db)):
    alerts = db.query(DataQualityAlert).order_by(DataQualityAlert.timestamp.desc()).limit(limit).all()
    return alerts

@app.get("/api/alerts/drift")
def get_drift_metrics(limit: int = 100, db: Session = Depends(get_db)):
    metrics = db.query(DriftMetric).order_by(DriftMetric.timestamp.desc()).limit(limit).all()
    return metrics

@app.get("/api/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Returns high-level stats for the dashboard.
    """
    latest_drift = db.query(DriftMetric).order_by(DriftMetric.timestamp.desc()).limit(20).all()
    
    # Calculate health based on recent activity (last 30 seconds)
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(seconds=30)
    recent_dq_count = db.query(DataQualityAlert).filter(DataQualityAlert.timestamp >= cutoff).count()
    
    # In 30 seconds at 100 events/sec, ~3,000 transactions are evaluated
    # Data Quality represents the % of successfully validated records
    estimated_window_events = 3000
    dq_health = max(0.0, min(100.0, 100.0 - ((recent_dq_count / estimated_window_events) * 100.0)))
    
    feature_status = {}
    for metric in latest_drift:
        if metric.field_name not in feature_status:
            feature_status[metric.field_name] = {
                'score': round(metric.score, 4),
                'status': 'Critical' if metric.severity == 'HIGH' else ('Warning' if metric.severity == 'MEDIUM' else 'Healthy')
            }
            
    drift_issues_count = sum(1 for m in latest_drift if m.is_drifting)
    
    schema_status = "Healthy" if recent_dq_count == 0 else ("Warning" if recent_dq_count < 150 else "Critical")
    
    return {
        "model_health": {
            "data_quality_percent": round(dq_health, 1),
            "feature_drift_alerts": drift_issues_count,
            "prediction_drift_alerts": 0,
            "freshness": "Healthy",
            "schema": schema_status
        },
        "feature_status": feature_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
