# Import necessary functions from preprocess_code.py
from preprocessing_code import validate_and_preprocess, prepare_training_data
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from model_utils import create_optimized_nn 
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import matplotlib.pyplot as plt

# Function to retrain the model from a file
def retrain_model_from_file(filepath):
    try:
        # Step 1: Preprocess Data using your existing preprocessing pipeline
        df, rating_mean = validate_and_preprocess(filepath)  # Validate and preprocess data
        preprocessors = joblib.load("saved_preprocessors/preprocessors_latest.pkl")  # Load preprocessor

        # Step 2: Prepare the training data
        processed_data = prepare_training_data(df)
        
        # Get the processed features and labels
        X_resampled = processed_data['X_resampled']
        y_resampled = processed_data['y_resampled']

        # Step 3: Create the model
        input_shape = (X_resampled.shape[1],)
        model = create_optimized_nn(input_shape=input_shape, num_classes=y_resampled.shape[1])

        # Step 4: Set up callbacks
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        # Step 5: Train the model
        history = model.fit(
            X_resampled, y_resampled,
            batch_size=16,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Step 6: Plot Training History (Optional)
        plot_training_history(history)

        # Step 7: Evaluate the model
        # Get predictions on validation data
        y_val_pred = model.predict(X_resampled)
        y_val_pred = np.argmax(y_val_pred, axis=1)
        y_val_true = np.argmax(y_resampled, axis=1)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_val_true, y_val_pred)
        f1 = f1_score(y_val_true, y_val_pred, average='weighted')
        recall = recall_score(y_val_true, y_val_pred, average='weighted')

        # Print metrics
        print(f"Evaluation after retraining:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")

        # Confusion Matrix (Optional)
        plot_confusion_matrix(y_val_true, y_val_pred)

        # Step 8: Save the retrained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"retrained_model_{timestamp}.h5"
        model.save(model_filename)
        
        # Optionally, save the updated LabelEncoder and preprocessors
        joblib.dump(preprocessors, f'saved_preprocessors/preprocessors_{timestamp}.pkl')
        
        # Return metrics for potential frontend usage
        return accuracy, f1, recall

    except Exception as e:
        print(f"Error during retraining: {e}")
        return None, None, None
