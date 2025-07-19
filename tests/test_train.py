import pytest
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Added for train-test split


# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


from utils import load_config, load_data
from train import train_model


class TestTrainingPipeline:
   
    def test_config_loading(self):
        """Test that configuration file loads successfully"""
        # Corrected path to config file
        config = load_config('config/config.json')
       
        # Check that config is loaded
        assert config is not None
        assert isinstance(config, dict)
       
        # Check required hyperparameters exist
        required_params = ['C', 'solver', 'max_iter']
        for param in required_params:
            assert param in config, f"Required parameter {param} missing from config"
       
        # Check data types
        assert isinstance(config['C'], (int, float)), "C should be numeric"
        assert isinstance(config['solver'], str), "solver should be string"
        assert isinstance(config['max_iter'], int), "max_iter should be integer"
   
    def test_model_creation(self):
        """Test model creation and training"""
        # Corrected path to config file
        config = load_config('config/config.json')
        X, y = load_data()
       
        # Train model
        model = train_model(X, y, config)
       
        # Check that model is LogisticRegression object
        assert isinstance(model, LogisticRegression), "Model should be LogisticRegression instance"
       
        # Check that model has been fitted
        assert hasattr(model, 'coef_'), "Model should have coef_ attribute after fitting"
        assert hasattr(model, 'classes_'), "Model should have classes_ attribute after fitting"
       
        # Check that classes are correct (0-9 digits)
        expected_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(model.classes_, expected_classes)
   
    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy on a test set"""
        # Corrected path to config file
        config = load_config('config/config.json')
        X_full, y_full = load_data()
       
        # Split data into training and testing sets for objective evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=config.get('random_state', 42)
        )
       
        # Train model on the training data only
        model = train_model(X_train, y_train, config)
       
        # Test accuracy on the test data (unseen data)
        accuracy = model.score(X_test, y_test)
       
        # Check that accuracy is above a reasonable threshold for a test set
        assert accuracy > 0.9, f"Model accuracy {accuracy:.4f} is below acceptable threshold of 0.9"
       
        # Test that model can make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), "Predictions length should match input length"
        assert all(pred in range(10) for pred in predictions), "All predictions should be digits 0-9"
   
    def test_data_loading(self):
        """Test data loading functionality"""
        X, y = load_data()
       
        # Check data shapes
        assert X.shape == (1797, 64), f"Expected X shape (1797, 64), got {X.shape}"
        assert y.shape == (1797,), f"Expected y shape (1797,), got {y.shape}"
       
        # Check data types
        assert X.dtype == np.float64, "X should be float64"
        assert y.dtype in [np.int32, np.int64], "y should be integer type"
       
        # Check label range
        assert set(y) == set(range(10)), "Labels should be digits 0-9"