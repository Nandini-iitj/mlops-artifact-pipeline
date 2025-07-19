import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split # Added for train-test split
from utils import load_config, load_data, save_model

def train_model(X_train, y_train, config):
    """Train logistic regression model with given configuration"""
    # The model is trained only on the training data
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config.get('random_state', 42)
    )
    
    model.fit(X_train, y_train)
    return model

def main():
    # Load configuration
    config = load_config('../config/config.json')

    # Load full dataset
    X_full, y_full = load_data()

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=config.get('random_state', 42)
    )
    
    # Train model on the training data
    print("Training model...")
    model = train_model(X_train, y_train, config)
    
    # Evaluate model on the training data (optional, but good for diagnostics)
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions, average='weighted')
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")

    # Evaluate model on the unseen test data
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='weighted')
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # Save model
    save_model(model, 'model_train.pkl')
    print("Model saved as model_train.pkl")

if __name__ == "__main__":
    main()