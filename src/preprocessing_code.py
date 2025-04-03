import pandas as pd
import numpy as np
import joblib
import os
import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable column names
NUMERICAL_COLS = ['latitude', 'longitude']
CATEGORICAL_COLS = ['care_system', 'mode of payment', 'Subcounty']
TEXT_COLS = ['services', 'operating_hours']

def validate_and_preprocess(data, prediction_mode=False):
    try:
        # Convert input data to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data.get('train', data))
        else:
            df = data.copy()
            df.columns = df.columns.astype(str)

        # Required columns
        required_columns = NUMERICAL_COLS + CATEGORICAL_COLS + TEXT_COLS
        if not prediction_mode:
            required_columns.append('rating')

        # Validate required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle missing ratings (only for training)
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            rating_mean = df['rating'].mean()
            df['rating'] = df['rating'].fillna(rating_mean)
        else:
            rating_mean = None

        return df, rating_mean

    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise


def prepare_categorical_features(df, preprocessors):
    try:
        if 'categorical_encoder' not in preprocessors or preprocessors['categorical_encoder'] is None:
            preprocessors['categorical_encoder'] = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = preprocessors['categorical_encoder'].fit_transform(df[CATEGORICAL_COLS])
        else:
            encoded_data = preprocessors['categorical_encoder'].transform(df[CATEGORICAL_COLS])

        # Generate feature names
        feature_names = []
        for i, col in enumerate(CATEGORICAL_COLS):
            categories = preprocessors['categorical_encoder'].categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])

        return pd.DataFrame(encoded_data, columns=feature_names)

    except Exception as e:
        logger.error(f"Categorical feature error: {str(e)}")
        raise


def prepare_text_features(df, preprocessors, max_features=100):
    try:
        if 'text_processors' not in preprocessors:
            preprocessors['text_processors'] = {}

        features = []
        feature_names = []

        for col in TEXT_COLS:
            if col not in df.columns:
                continue

            if col not in preprocessors['text_processors']:
                preprocessors['text_processors'][col] = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                text_features = preprocessors['text_processors'][col].fit_transform(df[col].fillna(''))
            else:
                text_features = preprocessors['text_processors'][col].transform(df[col].fillna(''))

            features.append(text_features.toarray())
            feature_names.extend([f"{col}_tfidf_{i}" for i in range(text_features.shape[1])])

        combined_features = np.hstack(features) if features else np.array([])

        return {
            'features': combined_features,
            'feature_names': feature_names
        }

    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        raise


def prepare_training_data(df, text_max_features=100):
    try:
        preprocessors = {
            'text_processors': {},
            'categorical_encoder': None,
            'scaler': None,
            'label_encoder': None,
            'rating_bins': None,
            'labels': None
        }
        df.columns = df.columns.astype(str)

        # Process categorical and text features
        categorical_encoded = prepare_categorical_features(df, preprocessors)
        text_features = prepare_text_features(df, preprocessors, text_max_features)

        # Process numerical features
        num_data = df[NUMERICAL_COLS].fillna(0).values

        # Combine all features
        X = np.hstack([num_data, categorical_encoded.values, text_features['features']])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Store feature names for consistency
        feature_names = NUMERICAL_COLS + categorical_encoded.columns.tolist() + text_features['feature_names']
        preprocessors.update({'scaler': scaler, 'feature_names': feature_names})

        # Encode labels
        label_encoder = LabelEncoder()
        rating_bins = [-float('inf'), 2.5, 3.5, float('inf')]
        labels = ['Low', 'Medium', 'High']
        df['rating_category'] = pd.cut(df['rating'], bins=rating_bins, labels=labels)
        y_encoded = label_encoder.fit_transform(df['rating_category'])

        preprocessors.update({
            'label_encoder': label_encoder,
            'rating_bins': rating_bins,
            'labels': labels
        })

        # Handle class imbalance
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_scaled, y_encoded)

        # Save preprocessors
        os.makedirs("saved_preprocessors", exist_ok=True)
        preprocessor_save_path = os.path.join("saved_preprocessors", "preprocessors_latest.pkl")
        joblib.dump(preprocessors, preprocessor_save_path)
        logger.info(f"Preprocessors saved successfully at: {preprocessor_save_path}")

        return {
            'X': X_scaled,
            'y': to_categorical(y_encoded),
            'X_resampled': X_resampled,
            'y_resampled': to_categorical(y_resampled),
            'preprocessors': preprocessors,
            'y_cat': y_encoded
        }

    except Exception as e:
        logger.error(f"Training data preparation failed: {str(e)}")
        raise


def prepare_prediction_data(input_data, preprocessors):
    try:
        # Convert input to DataFrame and ensure column names are strings
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
        df.columns = df.columns.astype(str)

        # Validate preprocessors
        if not preprocessors or not all(key in preprocessors for key in ['categorical_encoder', 'text_processors', 'scaler']):
            raise ValueError("Preprocessors dictionary is incomplete. Ensure training was completed successfully.")

        # Process numerical features
        num_data = df[NUMERICAL_COLS].fillna(0)

        # Process categorical features
        cat_data = preprocessors['categorical_encoder'].transform(df[CATEGORICAL_COLS])
        cat_df = pd.DataFrame(cat_data, columns=preprocessors['categorical_encoder'].get_feature_names_out())

        # Process text features
        text_features = []
        for col, vectorizer in preprocessors['text_processors'].items():
            if col in df.columns:
                tfidf_data = vectorizer.transform(df[col].fillna(''))
                text_features.append(pd.DataFrame(tfidf_data.toarray()))
            else:
                raise ValueError(f"Missing text column: {col}")

        # Combine all features
        X = pd.concat([num_data, cat_df] + text_features, axis=1)

        # Align features with training data
        if 'feature_names' not in preprocessors or not preprocessors['feature_names']:
            raise ValueError("Feature names missing. Ensure training was completed successfully.")

        X = X.reindex(columns=preprocessors['feature_names'], fill_value=0)

        # Apply scaling
        scaled_X = preprocessors['scaler'].transform(X)

        return scaled_X

    except Exception as e:
        logger.error(f"Prediction data preparation failed: {str(e)}")
        raise
