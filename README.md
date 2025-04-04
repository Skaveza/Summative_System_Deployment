#!/bin/bash
# HealthBridge - Healthcare Facility Rating Predictor
# MLOps Pipeline for Ugandan Healthcare Facilities

################################################################
### 🔍 PROJECT OVERVIEW
################################################################
# A Flask-based prediction system that classifies healthcare facility 
# quality (Low/Medium/High) using:
# - Facility services
# - Geographic coordinates  
# - Operating hours
# - Payment methods
# - Subcounty location

# Accuracy: 87% (tested on 3,396 Ugandan facilities)

################################################################
### QUICK START
################################################################

# Clone repository
git clone https://github.com/yourusername/HealthBridge.git
cd HealthBridge

# Install dependencies
pip3 install -r requirements.txt

# Start Flask server
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

################################################################
### 🌐 FLASK API ENDPOINTS
################################################################

#PREDICTION ENDPOINT
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

# Sample response:
# {
#   "prediction": "Medium",
#   "confidence": 0.89,
#   "probabilities": [0.08, 0.89, 0.03],
#   "timestamp": "2023-11-20T09:30:00Z"
# }

# RETRAINING ENDPOINT 
curl -X POST http://localhost:5000/api/retrain \
  -F "file=@new_facilities.csv" \
  -H "Authorization: Bearer YOUR_API_KEY"

################################################################
### DOCKER DEPLOYMENT
################################################################

# Build image
docker build -t healthbridge .

# Run container
docker run -p 5000:5000 healthbridge

################################################################
###MODEL PERFORMANCE
################################################################

# Classification Report
echo "+---------+-----------+--------+----------+----------------+"
echo "| Rating  | Precision | Recall | F1-Score | Facility Count |"
echo "+---------+-----------+--------+----------+----------------+"
echo "| Low     | 0.93      | 0.80   | 0.86     | 1,148          |"
echo "| Medium  | 0.85      | 0.96   | 0.90     | 1,120          |"
echo "| High    | 0.85      | 0.87   | 0.86     | 1,128          |"
echo "+---------+-----------+--------+----------+----------------+"
echo "| Overall Accuracy: 87% | Test Facilities: 3,396         |"
echo "+--------------------------------------------------------+"


################################################################
### LICENSE
################################################################

# MIT License
# Copyright (c) 2025 Sifa Kaveza Mwachoni
# See LICENSE file for details

echo "System ready! Access web interface at http://localhost:5000"
