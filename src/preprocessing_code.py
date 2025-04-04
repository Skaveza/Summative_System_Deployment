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

            # Add column prefix to make names unique
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
        X_scaled = scaler.fit_transform(pd.DataFrame(X, columns=feature_names))

        # Store feature names for consistency
        feature_names = NUMERICAL_COLS.copy()  # Make sure it's a new list
        feature_names.extend(categorical_encoded.columns.tolist())
        feature_names.extend(text_features['feature_names'])
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
        logger.info(f"Feature names count: {len(preprocessors['feature_names'])}")
        logger.info(f"Unique feature names: {len(set(preprocessors['feature_names']))}")
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
        print("\n=== STARTING PREPROCESSING ===")
        print(f"Initial input type: {type(input_data)}")
        if isinstance(input_data, dict):
            print("Input is dictionary - converting to DataFrame")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
        df.columns = df.columns.astype(str)
        
        print("\n=== AFTER CREATING DATAFRAME ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"First row values: {df.iloc[0].tolist()}")

        # Validate preprocessors
        required_keys = ['categorical_encoder', 'text_processors', 'scaler', 'feature_names']
        if not preprocessors or not all(key in preprocessors for key in required_keys):
            raise ValueError("Incomplete preprocessors. Retrain the model.")

        print("\n=== PROCESSING NUMERICAL FEATURES ===")
        print(f"Numerical columns: {NUMERICAL_COLS}")
        num_data = df[NUMERICAL_COLS].fillna(0).values
        print(f"Numerical data shape: {num_data.shape}")
        print(f"Numerical data sample: {num_data}")

        print("\n=== PROCESSING CATEGORICAL FEATURES ===")
        print(f"Categorical columns: {CATEGORICAL_COLS}")
        cat_data = preprocessors['categorical_encoder'].transform(df[CATEGORICAL_COLS])
        print(f"Raw categorical data shape: {cat_data.shape}")
        
        cat_df = pd.DataFrame(cat_data, columns=preprocessors['categorical_encoder'].get_feature_names_out())
        print(f"Encoded categorical data shape: {cat_df.shape}")
        print(f"First 5 categorical features: {cat_df.iloc[0, :5].tolist()}")

        print("\n=== PROCESSING TEXT FEATURES ===")
        print(f"Text columns: {TEXT_COLS}")
        text_features = []
        for col, vectorizer in preprocessors['text_processors'].items():
            print(f"\nProcessing text column: {col}")
            if col not in df.columns:
                raise ValueError(f"Missing text column: {col}")
                
            tfidf_data = vectorizer.transform(df[col].fillna(''))
            print(f"TF-IDF sparse matrix shape: {tfidf_data.shape}")
            
            text_array = tfidf_data.toarray()
            print(f"Converted to dense array shape: {text_array.shape}")
            text_features.append(text_array)

        print("\n=== COMBINING ALL FEATURES ===")
        print(f"Number of text feature arrays: {len(text_features)}")
        print(f"Shapes before combining - num: {num_data.shape}, cat: {cat_df.values.shape}, text: {[t.shape for t in text_features]}")
        
        X = np.hstack([num_data, cat_df.values] + text_features)
        print(f"Combined array shape: {X.shape}")
        
        print("\n=== SHAPE ADJUSTMENT ===")
        print(f"Current array dimensions: {X.ndim}")
        if X.ndim == 1:
            print("Reshaping from 1D to 2D")
            X = X.reshape(1, -1)
        elif X.ndim == 3:
            print("Reshaping from 3D to 2D")
            X = X.reshape(X.shape[0], -1)
        print(f"Shape after adjustment: {X.shape}")

        print("\n=== FEATURE ALIGNMENT ===")
        print(f"Number of expected features: {len(preprocessors['feature_names'])}")
        X_aligned = np.zeros((1, len(preprocessors['feature_names'])))
        print(f"Initial aligned array shape: {X_aligned.shape}")
        
        # This is a simplified alignment - you'll need to implement proper feature matching
        for i, col in enumerate(preprocessors['feature_names']):
            # Implement your actual feature matching logic here
            pass
            
        print(f"Aligned array shape: {X_aligned.shape}")

        print("\n=== SCALING ===")
        print(f"Input to scaler shape: {X_aligned.shape}")
        scaled_X = preprocessors['scaler'].transform(X_aligned)
        print(f"Scaled output shape: {scaled_X.shape}")

        print("\n=== FINAL SHAPE VALIDATION ===")
        if scaled_X.ndim != 2:
            print(f"Unexpected dimensions ({scaled_X.ndim}D), reshaping to 2D")
            scaled_X = scaled_X.reshape(1, -1)
        print(f"Final output shape: {scaled_X.shape}")
        print("\n=== PREPROCESSING FINAL CHECK ===")
        print(f"Returning array with shape: {scaled_X.shape}")
        print(f"Data type: {scaled_X.dtype}")
        print(f"Sample values: {scaled_X[0, :5]}")  # First 5 features
        return scaled_X

    except Exception as e:
        print("\n!!! ERROR DURING PREPROCESSING !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        if 'X' in locals():
            print(f"Last X shape: {X.shape if hasattr(X, 'shape') else 'No shape'}")
        if 'scaled_X' in locals():
            print(f"Last scaled_X shape: {scaled_X.shape if hasattr(scaled_X, 'shape') else 'No shape'}")
        raise