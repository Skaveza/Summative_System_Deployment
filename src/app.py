from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import time
import shutil
from retrain import retrain_model_from_file
import json

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')

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
                accuracy, f1, recall = retrain_model_from_file(file_path)
                
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
            form_data = {
                'facility_name': request.form.get('facility_name'),
                'services': request.form.get('services'),
                'latitude': float(request.form.get('latitude')),
                'longitude': float(request.form.get('longitude')),
                'operating_hours': request.form.get('operating_hours'),
                'care_system': request.form.get('care_system'),
                'mode of payment': request.form.get('payment_method'),
                'Subcounty': request.form.get('subcounty')
            }
            
            result = predictor.predict(form_data)
            
            if 'error' in result:
                flash(result['error'], 'error')
                return redirect(url_for('predict'))
            
            return render_template('predict.html',
                               prediction=result['prediction'],
                               probabilities=result['probabilities'],
                               confidence=result['confidence'],
                               form_data=form_data)
            
        except Exception as e:
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
            retrain_model_from_file(file_path)
            
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
