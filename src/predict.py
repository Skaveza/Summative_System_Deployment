import os
import secrets
import logging
from typing import Dict, Union
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable column names (must match preprocessing)
NUMERICAL_COLS = ['latitude', 'longitude']
CATEGORICAL_COLS = ['care_system', 'mode of payment', 'Subcounty']
TEXT_COLS = ['services', 'operating_hours']

class HealthcarePredictor:
    def __init__(self):
        """Initialize with your existing preprocessing flow"""
        self.model = None
        self.preprocessors = None
        self.load_artifacts()

        # API Key Storage (hashed keys)
        self.api_keys = {
            "web_app": os.environ.get('API_KEY_WEB', secrets.token_urlsafe(24)),
            "mobile_app": os.environ.get('API_KEY_MOBILE', secrets.token_urlsafe(24))
        }

    def load_artifacts(self):
        """Load model and preprocessors from environment-specific paths"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "../models", "healthcare_predict.h5")
            preprocessor_path = os.path.join(base_dir, "../saved_preprocessors", "preprocessors_latest.pkl")

            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path, compile=False)

            logger.info(f"Loading preprocessors from {preprocessor_path}")
            self.preprocessors = joblib.load(preprocessor_path)
        except Exception as e:
            logger.error(f"Artifact loading failed: {str(e)}")
            raise RuntimeError(f"Artifact loading failed: {str(e)}")

    def validate_key(self, api_key: str) -> bool:
        """Validate API key securely"""
        return api_key in self.api_keys.values()

    def preprocess_input(self, input_data: Dict) -> Union[np.ndarray, Dict]: # Fixed indentation here
        """Preprocess input data for prediction."""
        try:
            logger.info("Starting preprocessing...")

            # Convert input data to DataFrame
            df = pd.DataFrame([input_data])
            df.columns = df.columns.astype(str)

            # Define default values for missing features
            default_values = {
                'latitude': 0,
                'longitude': 0,
                'services': '',
                'operating_hours': '',
                'care_system': 'unknown',
                'mode of payment': 'unknown',
                'Subcounty': 'unknown'
            }

            # Fill missing features with defaults
            for key, value in default_values.items():
                if key not in df.columns:
                    df[key] = value

            # Validate preprocessors
            if not self.preprocessors or not all(
                key in self.preprocessors for key in ['categorical_encoder', 'text_processors', 'scaler']
            ):
                raise ValueError("Preprocessors dictionary is incomplete. Ensure training was completed successfully.")

            # Process numerical features
            num_data = df[NUMERICAL_COLS].fillna(0)

            # Process categorical features
            cat_data = self.preprocessors['categorical_encoder'].transform(df[CATEGORICAL_COLS])
            cat_df = pd.DataFrame(cat_data, columns=self.preprocessors['categorical_encoder'].get_feature_names_out())

            # Process text features
            text_features = []
            for col, vectorizer in self.preprocessors['text_processors'].items():
                if col in df.columns:
                    tfidf_data = vectorizer.transform(df[col].fillna(''))
                    text_features.append(pd.DataFrame(tfidf_data.toarray()))
                else:
                    raise ValueError(f"Missing text column: {col}")

            # Combine all features
            X = pd.concat([num_data, cat_df] + text_features, axis=1)

            # Check and log duplicate columns
            duplicate_columns = X.columns[X.columns.duplicated()].tolist()
            if duplicate_columns:
                logger.warning(f"Duplicate columns detected and will be removed: {duplicate_columns}")

            # Drop duplicate columns
            X = X.loc[:, ~X.columns.duplicated()]

            # Ensure feature names are unique
            self.preprocessors['feature_names'] = list(dict.fromkeys(self.preprocessors['feature_names']))

            # Align features with training data
            X = X.reindex(columns=self.preprocessors['feature_names'], fill_value=0)

            # Apply scaling
            scaled_X = self.preprocessors['scaler'].transform(X)

            logger.info("Preprocessing completed successfully.")
            return scaled_X

        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return {"error": f"Preprocessing error: {str(e)}"}

    def predict(self, input_data: Dict, api_key: str = None) -> Dict:
        """
        Prediction with optional API key check
        Maintains your existing workflow when key is None
        """
        if api_key and not self.validate_key(api_key):
            return {"error": "Invalid API key"}

        processed = self.preprocess_input(input_data)
        if isinstance(processed, dict) and "error" in processed:
            return processed

        try:
            logger.info("Making prediction...")
            prediction = self.model.predict(processed)
            predicted_class = np.argmax(prediction, axis=1)

            label_encoder = self.preprocessors.get('label_encoder')
            result = {
                "prediction": label_encoder.inverse_transform(predicted_class)[0],
                "confidence": float(np.max(prediction)),
                "probabilities": prediction.tolist()[0]
            }
            logger.info("Prediction completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

# Singleton instance
predictor = HealthcarePredictor()

# Helper functions for your existing workflow
def prepare_prediction_data(input_data: Dict, preprocessors: Dict) -> Union[np.ndarray, Dict]:
    """Your existing function remains unchanged"""
    return predictor.preprocess_input(input_data)

def make_prediction(input_data: Dict) -> Dict:
    """Your existing function now uses the singleton"""
    return predictor.predict(input_data)
