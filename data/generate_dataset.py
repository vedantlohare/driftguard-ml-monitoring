import pandas as pd
import numpy as np
import os

def generate_fraud_dataset(n_samples=100000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Generate base features
    user_age = np.random.normal(loc=35, scale=12, size=n_samples).astype(int)
    user_age = np.clip(user_age, 18, 90)
    
    user_income = np.random.lognormal(mean=11.0, sigma=0.6, size=n_samples)
    user_income = np.clip(user_income, 10000, 200000)
    
    transaction_amount = np.random.exponential(scale=50, size=n_samples)
    
    # 5 merchant categories
    merchant_category = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    
    distance_from_home = np.random.exponential(scale=10, size=n_samples)
    time_since_last_txn = np.random.exponential(scale=24, size=n_samples) # hours
    
    df = pd.DataFrame({
        'user_age': user_age,
        'user_income': user_income,
        'transaction_amount': transaction_amount,
        'merchant_category': merchant_category,
        'distance_from_home': distance_from_home,
        'time_since_last_txn': time_since_last_txn
    })
    
    # 2. Define the fraud logic (Concept)
    # Fraud is more likely if:
    # - transaction amount is unusually high
    # - distance from home is large
    # - time since last transaction is very small (rapid succession)
    
    fraud_prob = np.zeros(n_samples)
    
    # Base probability
    fraud_prob += 0.001
    
    # High amounts
    fraud_prob += np.where(df['transaction_amount'] > 200, 0.05, 0)
    fraud_prob += np.where(df['transaction_amount'] > 500, 0.15, 0)
    
    # Large distance
    fraud_prob += np.where(df['distance_from_home'] > 50, 0.05, 0)
    fraud_prob += np.where(df['distance_from_home'] > 200, 0.1, 0)
    
    # Rapid transactions
    fraud_prob += np.where(df['time_since_last_txn'] < 0.5, 0.1, 0)
    
    # Specific merchant categories (e.g. online electronics)
    fraud_prob += np.where(df['merchant_category'] == 4, 0.05, 0)
    
    # Cap probability
    fraud_prob = np.clip(fraud_prob, 0, 0.95)
    
    # Generate labels
    is_fraud = np.random.binomial(n=1, p=fraud_prob)
    df['is_fraud'] = is_fraud
    
    return df

if __name__ == "__main__":
    print("Generating baseline fraud dataset...")
    df = generate_fraud_dataset(n_samples=100000)
    
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_dataset.csv'), index=False)
    
    print(f"Dataset generated with {len(df)} records.")
    print(f"Fraud rate: {df['is_fraud'].mean():.4f}")
    print(df.head())
