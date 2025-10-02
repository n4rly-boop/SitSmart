import pickle
import numpy as np
from typing import Union

class Predictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)

    def _load_model(self, path: str):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def _load_scaler(self, path: str):
        try:
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded successfully from {path}")
            return scaler
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at {path}")
        except Exception as e:
            raise Exception(f"Error loading scaler: {str(e)}")

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_proba(self, X: Union[np.ndarray, list]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)

        return probabilities


if __name__ == "__main__":
    predictor = Predictor(
        model_path='../app/data/models/model.pkl',
        scaler_path='../app/data/models/scaler.pkl'
    )

    sample_data = np.array([[1, 2, 3, 4, 5]])

    prediction = predictor.predict(sample_data)
    print(f"Predicted class: {prediction[0]}")

    probabilities = predictor.predict_proba(sample_data)
    print(f"Class probabilities: {probabilities[0]}")
    print(f"Good posture probability: {probabilities[0][0]:.2%}")
    print(f"Bad posture probability: {probabilities[0][1]:.2%}")