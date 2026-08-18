import json
import logging
import os
import pandas as pd
from kafka import KafkaConsumer
from sqlalchemy.orm import Session
import time

from validator import DataValidator
from drift_engine import DriftEngine
from models import init_db, DataQualityAlert, DriftMetric

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'fraud_transactions')
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '500'))
DB_URL = os.getenv('DB_URL', 'postgresql://admin:password@localhost:5433/driftguard')

def get_db_session():
    # Retry mechanism for DB connection at startup
    retries = 5
    while retries > 0:
        try:
            SessionLocal = init_db(DB_URL)
            return SessionLocal()
        except Exception as e:
            logger.warning(f"Database connection failed, retrying... ({e})")
            retries -= 1
            time.sleep(5)
    raise Exception("Could not connect to Database after multiple retries.")

def main():
    logger.info("Initializing Stream Processor...")
    
    # 1. Init DB
    db: Session = get_db_session()
    logger.info("Database connected.")
    
    # 2. Init Validator
    validator = DataValidator()
    
    # 3. Init Drift Engine
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    baseline_path = os.path.join(data_dir, 'baseline_dataset.csv')
    if not os.path.exists(baseline_path):
        logger.error(f"Baseline dataset not found at {baseline_path}")
        return
        
    # We take a sample to keep memory reasonable
    logger.info("Loading baseline dataset...")
    baseline_df = pd.read_csv(baseline_path).sample(n=min(10000, sum(1 for _ in open(baseline_path)) - 1), random_state=42)
    drift_engine = DriftEngine(baseline_df)
    
    # 4. Init Kafka Consumer
    logger.info(f"Connecting to Kafka at {KAFKA_BROKER}...")
    # Add a small delay to ensure Kafka is fully up if started concurrently
    time.sleep(10)
    
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    logger.info("Kafka consumer connected.")
    
    buffer = []
    
    logger.info("Listening for events...")
    for message in consumer:
        record = message.value
        
        # 1. Validate Data Quality
        val_result = validator.validate_record(record)
        
        if not val_result['is_valid']:
            # Log Data Quality Issue
            for violation in val_result['violations']:
                alert = DataQualityAlert(
                    severity='HIGH',
                    issue_type='SCHEMA_OR_BOUND',
                    description=violation,
                    raw_data=record
                )
                db.add(alert)
            db.commit()
            continue # Skip adding invalid records to drift buffer
            
        # 2. Add to buffer for drift detection
        buffer.append(record)
        
        # 3. Check for drift if buffer is full
        if len(buffer) >= WINDOW_SIZE:
            logger.info(f"Window full ({WINDOW_SIZE} records). Running drift detection...")
            window_df = pd.DataFrame(buffer)
            
            drift_results = drift_engine.detect_drift(window_df)
            
            for field, res in drift_results.items():
                if res['drift_detected']:
                    logger.warning(f"Drift detected on {field}: {res}")
                
                metric = DriftMetric(
                    field_name=field,
                    metric_type='PSI' if res['type'] == 'categorical' else 'KS',
                    score=res.get('psi', res.get('ks_statistic')),
                    is_drifting=res['drift_detected'],
                    severity=res['severity']
                )
                db.add(metric)
                
            db.commit()
            buffer = [] # Clear buffer

if __name__ == "__main__":
    main()
