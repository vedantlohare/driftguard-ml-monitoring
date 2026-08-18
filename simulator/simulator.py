import json
import time
import random
import numpy as np
from kafka import KafkaProducer
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'fraud_transactions')

def create_producer():
    return KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

def generate_event(mode="healthy"):
    """
    Generates a single transaction event.
    If mode == 'drift', we shift the distributions.
    If mode == 'quality_issue', we inject schema violations.
    """
    if mode == "quality_issue":
        # Inject bad data
        return {
            'user_age': random.choice([-10, 150, None]), # Out of bounds or missing
            'user_income': -500, # Invalid
            'transaction_amount': "one hundred", # Schema violation
            'merchant_category': 99, # Invalid category
            'distance_from_home': 10.0,
            # missing time_since_last_txn
        }
        
    if mode == "drift":
        # Feature drift: older users, much higher amounts, different merchants
        age = int(np.clip(np.random.normal(loc=55, scale=10), 18, 90))
        income = float(np.random.lognormal(mean=12.0, sigma=0.8))
        amount = float(np.random.exponential(scale=200))
        cat = int(np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.1, 0.2, 0.5]))
    else:
        # Healthy (similar to baseline - strictly valid schema and bounds)
        age = int(np.clip(np.random.normal(loc=35, scale=12), 18, 90))
        income = float(np.clip(np.random.lognormal(mean=11.0, sigma=0.6), 10000, 200000))
        amount = float(np.random.exponential(scale=50))
        cat = int(np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.2, 0.15, 0.15, 0.1]))
        
    return {
        'user_age': age,
        'user_income': max(0.0, income),
        'transaction_amount': max(0.01, amount),
        'merchant_category': cat,
        'distance_from_home': max(0.0, float(np.random.exponential(scale=10))),
        'time_since_last_txn': max(0.0, float(np.random.exponential(scale=24)))
    }

def main():
    logger.info(f"Connecting to Kafka at {KAFKA_BROKER}...")
    producer = create_producer()
    logger.info("Connected!")
    
    events_sent = 0
    
    try:
        while True:
            # Cycle through modes every 4000 events
            cycle_step = events_sent % 4000
            
            if cycle_step < 1000:
                mode = "healthy"
            elif cycle_step < 1100:
                mode = "quality_issue"
            elif cycle_step < 2000:
                mode = "healthy"
            elif cycle_step < 3500:
                mode = "drift"
            else:
                mode = "healthy"
                
            event = generate_event(mode)
            producer.send(KAFKA_TOPIC, value=event)
            events_sent += 1
            
            if events_sent % 200 == 0:
                logger.info(f"Sent {events_sent} events. Current mode: {mode} (Cycle step: {cycle_step})")
                
            time.sleep(0.01) # 100 events per second
            
    except KeyboardInterrupt:
        logger.info("Stopping simulator.")
    finally:
        producer.close()

if __name__ == "__main__":
    # Wait for Kafka to be ready
    time.sleep(5)
    main()
