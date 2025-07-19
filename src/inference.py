import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_data, load_model

def make_predictions(model, X):
    """Make predictions using the trained model"""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities

def main():
    try:
        # Load the trained model
        print("Loading trained model...")
        model = load_model('model_train.pkl')
        
        # Load the dataset
        print("Loading dataset...")
        X, y = load_data()
        
        # Make predictions
        print("Making predictions...")
        predictions, probabilities = make_predictions(model, X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')
        
        print(f"\nInference Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Print detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y, predictions))
        
        # Print some example predictions
        print(f"\nSample Predictions (first 10):")
        for i in range(min(10, len(predictions))):
            confidence = np.max(probabilities[i])
            print(f"True: {y[i]}, Predicted: {predictions[i]}, Confidence: {confidence:.3f}")
        
        print(f"\nInference completed successfully!")
        
    except FileNotFoundError:
        print("Error: model_train.pkl not found. Please run training first.")
        raise
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()