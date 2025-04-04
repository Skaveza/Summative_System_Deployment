from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import time
import shutil
from retrain import train_model
from predict import predictor
import json
import numpy as np
from preprocessing_code import prepare_prediction_data
import joblib


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')

preprocessor_save_path = os.path.join("../saved_preprocessors", "preprocessors_latest.pkl")
preprocessors = joblib.load(preprocessor_save_path)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for all HTML pages
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return redirect(url_for('home'))

@app.route('/preprocess')
def preprocess():
    return render_template('preprocess.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['dataset']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                accuracy, f1, recall = train_model(file_path)
                
                if accuracy is not None:
                    flash(f'Model retrained successfully! Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}', 'success')
                else:
                    flash('Retraining failed. Please try again.', 'error')
                
                if os.path.exists(file_path):
                    os.remove(file_path)

            except Exception as e:
                flash(f'Retraining failed: {str(e)}', 'error')
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return redirect(url_for('retrain'))
        else:
            flash('Invalid file format. Only CSV files are allowed.', 'error')
            return redirect(url_for('retrain'))
    
    return render_template('retrain.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'services': request.form.get('services'),
                'latitude': float(request.form.get('latitude')),
                'longitude': float(request.form.get('longitude')),
                'operating_hours': request.form.get('operating_hours'),
                'care_system': request.form.get('care_system'),
                'mode of payment': request.form.get('Payment'),
                'Subcounty': request.form.get('Subcounty')
            }
            
            # Preprocess the data
            input_data = prepare_prediction_data(form_data, preprocessors)
            
            # DEBUG: Print the raw preprocessed data
            print(f"\n=== PREPROCESSED DATA ===")
            print(f"Type: {type(input_data)}")
            print(f"Shape: {input_data.shape}")
            print(f"Data type: {input_data.dtype}")
            print(f"First 5 values: {input_data[0, :5]}")
            
            # Ensure proper shape and type
            if isinstance(input_data, np.ndarray):
                if input_data.ndim == 3:
                    print("Reshaping from 3D to 2D")
                    input_data = input_data.reshape(input_data.shape[0], -1)
                elif input_data.ndim == 1:
                    print("Reshaping from 1D to 2D")
                    input_data = input_data.reshape(1, -1)
                    
                # Convert to float32 if needed
                if input_data.dtype != np.float32:
                    print(f"Converting dtype from {input_data.dtype} to float32")
                    input_data = input_data.astype(np.float32)
            else:
                print("Converting non-array input to numpy array")
                input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
            
            # Final shape verification
            print(f"\n=== FINAL PREDICTION INPUT ===")
            print(f"Shape: {input_data.shape}")
            print(f"Type: {type(input_data)}")
            print(f"Data type: {input_data.dtype}")
            
            # Make prediction
            predictions = predictor.predict(input_data)
            predictions = predictor.predict(input_data)
            
            # Process results
            if predictions is not None and len(predictions) > 0:
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions, axis=1)[0]
                
                return render_template('predict.html',
                                    prediction=int(predicted_class),
                                    probabilities=predictions[0].tolist(),
                                    confidence=float(confidence * 100),
                                    form_data=form_data)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            flash(f'Prediction error: {str(e)}', 'error')
            return redirect(url_for('predict'))
    
    return render_template('predict.html')



@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    if 'dataset' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['dataset']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            train_model(file_path)
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({"message": "Model retrained successfully!"}), 200
        
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Invalid file format. Only CSV files are allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True)
