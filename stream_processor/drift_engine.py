import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import logging
import json

logger = logging.getLogger(__name__)

class DriftEngine:
    def __init__(self, baseline_df: pd.DataFrame):
        """
        Initializes the drift engine with baseline data.
        In a real production system, this could just be a random sample of 10k rows 
        from the training set to keep memory low.
        """
        self.baseline_df = baseline_df
        
    def _calculate_psi(self, expected, actual, buckets=10):
        """
        Calculate the Population Stability Index (PSI) between two arrays.
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input
            
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid zero division
        def sub_zero(x):
            return np.where(x == 0, 0.0001, x)
            
        expected_percents = sub_zero(expected_percents)
        actual_percents = sub_zero(actual_percents)
        
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        return np.sum(psi_values)

    def detect_drift(self, window_df: pd.DataFrame):
        """
        Compares a window of recent production data against the baseline.
        Returns a dictionary of drift metrics per feature.
        """
        results = {}
        
        for col in window_df.columns:
            if col not in self.baseline_df.columns:
                continue
                
            base_col = self.baseline_df[col].dropna()
            win_col = window_df[col].dropna()
            
            if len(base_col) == 0 or len(win_col) == 0:
                continue
                
            # If categorical (e.g., merchant_category or integer values with few unique items)
            if base_col.nunique() < 20:
                # Use PSI for categorical
                base_counts = base_col.value_counts(normalize=True)
                win_counts = win_col.value_counts(normalize=True)
                
                # Align indices
                all_cats = set(base_counts.index).union(set(win_counts.index))
                base_freq = np.array([base_counts.get(c, 0.0001) for c in all_cats])
                win_freq = np.array([win_counts.get(c, 0.0001) for c in all_cats])
                
                psi = np.sum((win_freq - base_freq) * np.log(win_freq / base_freq))
                
                results[col] = {
                    'type': 'categorical',
                    'psi': float(psi),
                    'drift_detected': psi > 0.1,
                    'severity': 'HIGH' if psi > 0.2 else ('MEDIUM' if psi > 0.1 else 'LOW')
                }
            else:
                # Use KS test and PSI for continuous
                ks_stat, p_value = ks_2samp(base_col, win_col)
                psi = self._calculate_psi(base_col.values, win_col.values)
                
                results[col] = {
                    'type': 'continuous',
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'psi': float(psi),
                    'drift_detected': p_value < 0.05 or psi > 0.1,
                    'severity': 'HIGH' if (p_value < 0.01 or psi > 0.2) else ('MEDIUM' if (p_value < 0.05 or psi > 0.1) else 'LOW')
                }
                
        return results

if __name__ == "__main__":
    # Test the drift engine
    print("Testing Drift Engine...")
    base_data = pd.DataFrame({'age': np.random.normal(30, 5, 1000), 'category': np.random.choice([0,1,2], 1000)})
    
    # Healthy window
    healthy_win = pd.DataFrame({'age': np.random.normal(30, 5, 200), 'category': np.random.choice([0,1,2], 200)})
    
    # Drifted window (age shifted up, category skewed)
    drifted_win = pd.DataFrame({'age': np.random.normal(40, 5, 200), 'category': np.random.choice([0,1,2], 200, p=[0.1, 0.1, 0.8])})
    
    engine = DriftEngine(base_data)
    print("Healthy Window Drift Results:")
    print(json.dumps(engine.detect_drift(healthy_win), indent=2))
    
    print("\nDrifted Window Drift Results:")
    import json
    print(json.dumps(engine.detect_drift(drifted_win), indent=2))
