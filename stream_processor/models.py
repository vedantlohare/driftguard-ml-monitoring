from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class DataQualityAlert(Base):
    __tablename__ = 'data_quality_alerts'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    severity = Column(String) # LOW, MEDIUM, HIGH
    issue_type = Column(String) # SCHEMA, OUTLIER, MISSING
    field_name = Column(String, nullable=True)
    description = Column(String)
    raw_data = Column(JSON, nullable=True)

class DriftMetric(Base):
    __tablename__ = 'drift_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    field_name = Column(String, index=True)
    metric_type = Column(String) # PSI, KS
    score = Column(Float)
    is_drifting = Column(Boolean)
    severity = Column(String) # LOW, MEDIUM, HIGH

def init_db(database_url="postgresql://admin:password@localhost:5433/driftguard"):
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

if __name__ == "__main__":
    print("Initializing Database Schema...")
    # Will fail if postgres isn't running yet, just a test block
    try:
        init_db()
        print("Schema created successfully!")
    except Exception as e:
        print(f"Error initializing DB: {e}")
