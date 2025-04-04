# HealthBridge - Healthcare Facility Rating Predictor
## MLOps Pipeline for Ugandan Healthcare Facilities

## Demo Video - https://www.loom.com/share/6494f55dd0ad42c1938c1bccad9d6611?sid=d25238bc-4993-4382-9c48-d6c21ee88735

### Project Overview
HealthBridge is a user-friendly, **FastAPI-based** machine learning system that predicts the quality of healthcare facilities in Uganda. It classifies facilities as **Low, Medium, or High** quality based on:
- Services provided
- Geographic location (latitude & longitude)
- Operating hours
- Payment methods accepted
- Subcounty location

**Model Accuracy:** 87% 

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/HealthBridge.git
cd HealthBridge
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Start the FastAPI server
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```
Now, your API is up and running at **http://localhost:5000**!

---

## Using the API

### Predict Healthcare Facility Quality
Send a request with facility details to get a quality prediction:
```bash
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
```

#### Sample Response:
```json
{
  "prediction": "Medium",
  "confidence": 0.89,
  "probabilities": [0.08, 0.89, 0.03],
  "timestamp": "2023-11-20T09:30:00Z"
}
```

### Update Model with New Data
Upload a new dataset to retrain the model:
```bash
curl -X POST http://localhost:5000/api/retrain \ 
  -F "file=@new_facilities.csv" \ 
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Model Performance

### Classification Report:
```
+---------+-----------+--------+----------+----------------+
| Rating  | Precision | Recall | F1-Score | Facility Count |
+---------+-----------+--------+----------+----------------+
| Low     | 0.93      | 0.80   | 0.86     | 1,148          |
| Medium  | 0.85      | 0.96   | 0.90     | 1,120          |
| High    | 0.85      | 0.87   | 0.86     | 1,128          |
+---------+-----------+--------+----------+----------------+
| Overall Accuracy: 87% | Test Facilities: 3,396 |
+--------------------------------------------------------+
```

---

## Retraining Pipeline
1. **Upload new data** via UI or API endpoint (`/upload`).
2. **Trigger retraining** via the `/retrain` endpoint or UI button.
3. **Pipeline automatically:**
   - Processes the new data.
   - Retrains the model.
   - Saves & updates the deployed model.

---

## Performance Testing
- Simulated high-traffic requests using **Locust**.
- Measured latency and response times under different configurations.
- Results:
  - **Latency:** [XX] ms
  - **Response Time:** [XX] ms (with varying load conditions)
  - **Max Requests per Second:** [XX]

---

## Video Demo
Watch the full demo here: [YouTube Link]

---

## License

**MIT License**  
© 2025 Sifa Kaveza Mwachoni  


---

**System is ready!** Access the web interface at: [http://localhost:5000](http://localhost:5000)

