# HealthBridge - Healthcare Facility Rating Predictor
## MLOps Pipeline for Ugandan Healthcare Facilities

---

## **Video Demo**
Watch the full demo here: (https://www.loom.com/share/6494f55dd0ad42c1938c1bccad9d6611?sid=92e0d92c-2f54-42a5-aa37-399d83f2b8fe)
API Endpoint: http://127.0.0.1:5000/home

---

## **Project Overview**
HealthBridge is a **Flask-based machine learning system** that predicts the quality of healthcare facilities in Uganda. It classifies facilities as **Low, Medium, or High** quality based on:
- Services provided
- Geographic location (latitude & longitude)
- Operating hours
- Payment methods accepted
- Subcounty location

### **Model and Dataset**
- **Dataset**: The model was trained on the [Pollicy/Uganda_HealthCare_Facilities](https://huggingface.co/datasets/Pollicy/Uganda_HealthCare_Facilities) dataset, containing **6,000+** healthcare facilities.
- **Model Architecture**: Utilizes a **Neural Network Model** for classification.
- **Evaluation Metrics**: The model was assessed using **Accuracy**, **Precision**, **Recall**, and **F1-score**, achieving an overall **87% accuracy**.

---

## **Getting Started**

### 1. Clone the repository
```bash
git clone https://github.com/ySkaveza/Summative_System_Deployment.git
cd HealthBridge
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Start the FastAPI Server
```bash
gunicorn app:app --host 0.0.0.0 --port 8000
```
Now, your API is up and running at **http://localhost:8000**!

---

## **Using the API**

### **Predict Healthcare Facility Quality**
Send a request with facility details to get a quality prediction:
```bash
curl -X POST http://localhost:8000/api/predict \ 
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

#### **Sample Response:**
```json
{
  "prediction": "Medium",
  "confidence": 0.89,
  "probabilities": [0.08, 0.89, 0.03],
  "timestamp": "2023-11-20T09:30:00Z"
}
```

### **Update Model with New Data**
Upload a new dataset to retrain the model:
```bash
curl -X POST http://localhost:8000/api/retrain \ 
  -F "file=@new_facilities.csv" \ 
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## **Retraining Pipeline**
1. **Upload new data** via UI or API endpoint (`/upload`).
2. **Trigger retraining** using the `/retrain` endpoint or UI button.
3. **Pipeline automatically:**
   - Processes the new data.
   - Retrains the model.
   - Saves & updates the deployed model.

---

## **How to Contribute**
We welcome contributions! Follow these steps:
1. **Fork the repository** and clone it locally.
2. **Create a new branch** for your feature or bug fix.
3. **Commit your changes** with a clear message.
4. **Push to your fork** and submit a pull request.


## **Contributors**
- **Sifa Kaveza Mwachoni**

---

## **License**
This project is licensed under the **MIT License**. See (LICENSE) for details.

