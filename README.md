#!/bin/bash
# HealthBridge - Healthcare Facility Rating Predictor
# MLOps Pipeline for Ugandan Healthcare Facilities

################################################################
### PROJECT OVERVIEW
################################################################
# HealthBridge is a user-friendly, Flask-based machine learning system 
# that predicts the quality of healthcare facilities in Uganda. 
# It classifies facilities as Low, Medium, or High quality based on:
# - Services provided
# - Geographic location (latitude & longitude)
# - Operating hours
# - Payment methods accepted
# - Subcounty location
# 
# Model Accuracy: 87% (Tested on 3,396 Ugandan healthcare facilities)

################################################################
### GETTING STARTED
################################################################

# 1. Clone the repository
git clone https://github.com/yourusername/HealthBridge.git
cd HealthBridge

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Start the Flask server
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

Now, your API is up and running at http://localhost:5000!

################################################################
### USING THE API
################################################################

# PREDICT HEALTHCARE FACILITY QUALITY
# Send a request with facility details to get a quality prediction:
curl -X POST http://localhost:5000/api/predict \ 
  -H "Content-Type: application/json" \ 
  -d '{
    "services": "Outpatient, Maternity, Emergency",
    "latitude": 0.3136,
    "longitude": 32.5811,
    "operating_hours": "Mon-Fri 8AM-5PM",
    "care_system": "Public",
    "payment_method": "NHIF",
    "subcounty": "Kampala Central"
  }'

# Sample Response:
# {
#   "prediction": "Medium",
#   "confidence": 0.89,
#   "probabilities": [0.08, 0.89, 0.03],
#   "timestamp": "2023-11-20T09:30:00Z"
# }

# UPDATE MODEL WITH NEW DATA
# Upload a new dataset to retrain the model:
curl -X POST http://localhost:5000/api/retrain \ 
  -F "file=@new_facilities.csv" \ 
  -H "Authorization: Bearer YOUR_API_KEY"

################################################################
### DEPLOYING WITH DOCKER
################################################################

# 1. Build the Docker image
docker build -t healthbridge .

# 2. Run the container
docker run -p 5000:5000 healthbridge

Your app is now running inside a Docker container!

################################################################
### MODEL PERFORMANCE
################################################################

# Classification Report:
printf "\n+---------+-----------+--------+----------+----------------+"
printf "\n| Rating  | Precision | Recall | F1-Score | Facility Count |"
printf "\n+---------+-----------+--------+----------+----------------+"
printf "\n| Low     | 0.93      | 0.80   | 0.86     | 1,148          |"
printf "\n| Medium  | 0.85      | 0.96   | 0.90     | 1,120          |"
printf "\n| High    | 0.85      | 0.87   | 0.86     | 1,128          |"
printf "\n+---------+-----------+--------+----------+----------------+"
printf "\n| Overall Accuracy: 87% | Test Facilities: 3,396 |"
printf "\n+--------------------------------------------------------+\n"

################################################################
### LICENSE
################################################################

# MIT License
# © 2025 Sifa Kaveza Mwachoni
# See LICENSE file for details

echo "System is ready! Access the web interface at: http://localhost:5000"

