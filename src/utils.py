import json
import pickle
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_data():
    """Load and return digits dataset"""
    digits = load_digits()
    return digits.data, digits.target

def save_model(model, filepath):
    """Save model using joblib"""
    joblib.dump(model, filepath)
    
def load_model(filepath):
    """Load model using joblib"""
    return joblib.load(filepath)