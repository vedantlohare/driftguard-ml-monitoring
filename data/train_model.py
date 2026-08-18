import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def train_and_save_model():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(data_dir, 'baseline_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please run generate_dataset.py first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Save the model
    model_path = os.path.join(data_dir, 'fraud_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Calculate baseline statistics for drift monitoring
    print("Calculating baseline statistics for drift detection...")
    baseline_stats = {}
    for col in X.columns:
        if col == 'merchant_category':
            # categorical
            val_counts = X_train[col].value_counts(normalize=True).to_dict()
            # ensure all keys are strings for json
            baseline_stats[col] = {'type': 'categorical', 'distribution': {str(k): float(v) for k, v in val_counts.items()}}
        else:
            # continuous
            baseline_stats[col] = {
                'type': 'continuous',
                'mean': float(X_train[col].mean()),
                'std': float(X_train[col].std()),
                'min': float(X_train[col].min()),
                'max': float(X_train[col].max()),
                'quantiles': {
                    'q25': float(X_train[col].quantile(0.25)),
                    'q50': float(X_train[col].quantile(0.50)),
                    'q75': float(X_train[col].quantile(0.75))
                }
            }
            
    # Also save the distribution of predictions on the train set (for prediction drift)
    train_preds = model.predict_proba(X_train)[:, 1]
    baseline_stats['prediction'] = {
        'type': 'continuous',
        'mean': float(np.mean(train_preds)),
        'std': float(np.std(train_preds)),
        'quantiles': {
            'q25': float(np.quantile(train_preds, 0.25)),
            'q50': float(np.quantile(train_preds, 0.50)),
            'q75': float(np.quantile(train_preds, 0.75))
        }
    }
    
    stats_path = os.path.join(data_dir, 'baseline_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(baseline_stats, f, indent=4)
        
    print(f"Baseline statistics saved to {stats_path}")

if __name__ == "__main__":
    train_and_save_model()
