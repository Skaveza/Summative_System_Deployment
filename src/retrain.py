import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf
import matplotlib.pyplot as plt

# Import preprocessing functions
from preprocessing_code import validate_and_preprocess, prepare_training_data

def train_model(data_path, output_dir="models"):
    """
    Trains an optimized neural network model for classification.

    Parameters:
    - data_path (str): Path to the dataset file (CSV format).
    - output_dir (str): Directory where the trained model will be saved.

    Returns:
    - dict: Contains training history and model evaluation results.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load Dataset
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Step 2: Preprocess the data
    print("Preprocessing data...")
    validated_data, _ = validate_and_preprocess(df)
    prep_result = prepare_training_data(validated_data)

    # Extract components
    X_resampled = prep_result['X_resampled']
    y_resampled = prep_result['y_resampled']

    # Step 3: Split data into train, validation, and test sets
    print("Splitting data into train/validation/test sets...")

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    splits = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Compute class weights
    def get_class_weights(y_resampled):
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_resampled),
            y=y_resampled
        )
        return dict(enumerate(class_weights))

    class_weights = get_class_weights(np.argmax(y_train, axis=1))  # Convert one-hot labels to categorical

    # Step 4: Define the Neural Network Model
    def create_model(input_shape, num_classes=3):
        model = Sequential([
            Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.4),

            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.3),

            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.3),

            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.2),

            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    input_shape = (X_train.shape[1],)
    model = create_model(input_shape, num_classes=3)

    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    # Step 5: Train the model
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Step 6: Evaluate the model
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Step 7: Save the model
    model_filename = os.path.join(output_dir, "optimized_model.h5")
    model.save(model_filename)
    print(f"Model saved at: {model_filename}")

    # Step 8: Save preprocessors
    preprocessors_filename = os.path.join(output_dir, "preprocessors.pkl")
    joblib.dump(prep_result['preprocessors'], preprocessors_filename)
    print(f"Preprocessors saved at: {preprocessors_filename}")

    # Step 9: Plot Training History
    def plot_history(history):
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    plot_history(history)

    # Step 10: Return training results
    return {
        "train_samples": X_train.shape[0],
        "validation_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "model_path": model_filename,
        "preprocessors_path": preprocessors_filename
    }
