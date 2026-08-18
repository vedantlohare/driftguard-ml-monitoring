import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        # Define expected schema and rules
        self.expected_schema = {
            'user_age': {'type': (int, float), 'min': 18, 'max': 120},
            'user_income': {'type': (int, float), 'min': 0},
            'transaction_amount': {'type': (int, float), 'min': 0},
            'merchant_category': {'type': int, 'allowed_values': [0, 1, 2, 3, 4]},
            'distance_from_home': {'type': (int, float), 'min': 0},
            'time_since_last_txn': {'type': (int, float), 'min': 0}
        }
        
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a single record against the expected schema.
        Returns a dictionary with 'is_valid' boolean and 'violations' list.
        """
        violations = []
        
        # 1. Check for missing critical fields
        for field in self.expected_schema.keys():
            if field not in record or record[field] is None:
                violations.append(f"Missing required field: {field}")
                
        if violations:
            return {'is_valid': False, 'violations': violations}
            
        # 2. Check types, bounds, and allowed values
        for field, rules in self.expected_schema.items():
            val = record.get(field)
            
            # Type check
            if 'type' in rules and not isinstance(val, rules['type']):
                violations.append(f"Type violation on {field}: expected {rules['type']}, got {type(val)}")
                continue # Skip further checks for this field if type is wrong
                
            # Bounds check
            if 'min' in rules and val < rules['min']:
                violations.append(f"Bound violation on {field}: value {val} is less than min {rules['min']}")
            if 'max' in rules and val > rules['max']:
                violations.append(f"Bound violation on {field}: value {val} is greater than max {rules['max']}")
                
            # Allowed values check (Categorical)
            if 'allowed_values' in rules and val not in rules['allowed_values']:
                violations.append(f"Value violation on {field}: value {val} not in allowed {rules['allowed_values']}")
                
        return {
            'is_valid': len(violations) == 0,
            'violations': violations
        }

if __name__ == "__main__":
    validator = DataValidator()
    
    # Test valid
    valid_rec = {
        'user_age': 45, 'user_income': 50000.0, 'transaction_amount': 150.5,
        'merchant_category': 2, 'distance_from_home': 10.0, 'time_since_last_txn': 5.0
    }
    print(f"Valid record test: {validator.validate_record(valid_rec)}")
    
    # Test invalid schema
    invalid_rec = {
        'user_age': 15, # Too young
        'user_income': -100, # Negative
        'transaction_amount': "high", # Wrong type
        'merchant_category': 99, # Invalid category
        # Missing distance and time
    }
    print(f"Invalid record test: {json.dumps(validator.validate_record(invalid_rec), indent=2)}")
