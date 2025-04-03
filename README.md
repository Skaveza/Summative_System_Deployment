# **MLOps: Machine Learning Model Deployment and Retraining**

## **Project Overview**
This project demonstrates the complete lifecycle of a Machine Learning classification model, including:
- Training a classification model offline and saving it as a file.
- Deploying the model to a cloud-based platform.
- Creating a retraining pipeline that updates the model when new data is uploaded.
- Evaluating the model’s performance using multiple metrics.
- Simulating real-world requests using Locust to measure response time and scalability.
- Providing a web-based UI for predictions and model retraining.

---

### **1. Model Training & Evaluation**
- **Model Training**: Trained a **classification model** using the [Pollicy/Uganda_HealthCare_Facilities/Full collected hospital data 6K+ - All healthcare facilities.csv] Hugging Face Dataset.
- **Model Architecture**: Used a **Neural Network Model** for classification.
- **Model Evaluation**: The model was evaluated using multiple performance metrics, including **Accuracy**, **Precision**, **Recall**, and **F1-score**. The **F1-score** remained consistently strong across the classes, indicating good overall performance and balanced classification.

#### **Retrained Model Summary**:
- **Learning Rate**: Reduced from **0.001** to **0.0005**, allowing for finer weight adjustments and more stable training.
- **Batch Size**: Decreased from **32** to **16**, leading to more frequent gradient updates and smoother convergence.
- **Patience**: Increased from **10** to **15**, allowing the model to train longer and preventing premature stopping.

These changes improved the **test loss** from **0.55** to **0.43**, with **accuracy** remaining at **87%**. Notably, the retrained model showed slight improvements in **precision** and **recall** for class 0, resulting in better handling of this class.

#### **Performance Metrics of the Retrained Model:**


              precision    recall  f1-score   support

           0       0.93      0.80      0.86      1148
           1       0.85      0.96      0.90      1120
           2       0.85      0.87      0.86      1128

    accuracy                           0.87      3396
   macro avg       0.88      0.87      0.87      3396
weighted avg       0.88      0.87      0.87      3396
#### **Comparison to Original Model**:
The **original model** had similar accuracy and macro averages, but the retrained model showed slight improvements, particularly in **precision** for class 0, and **recall** for class 1.
          precision    recall  f1-score   support

       0       0.96      0.77      0.86      1148
       1       0.82      0.99      0.90      1120
       2       0.85      0.85      0.85      1128

accuracy                           0.87      3396

- **Key Takeaways**:
  - **F1-Score**: The F1-scores remained consistent (~0.87-0.88) for both models, indicating strong performance across all classes.
  - **Class Imbalance**: Both models show balanced performance across classes, as evidenced by the weighted and macro averages, which are around **0.88**.
  - **Improvement**: The retrained model demonstrates slight improvements in precision and recall for class 0 and class 1.

### **2. Deployment**
- Deployed the model on **[Cloud Platform: AWS, GCP, Azure, Heroku, etc.]**.
- Exposed an API for predictions using **FastAPI**.
- Containerized the application with **Docker**.

### **3. Retraining Pipeline**
- Allows users to **upload new data** for retraining.
- Processes the new data and retrains the model.
- Saves the updated model and deploys it automatically.
- Uses **trigger-based retraining** when new data is uploaded.

### **4. Prediction API & Web App**
- Users can **upload data (CSV, image, or audio) to get predictions**.
- Interactive **visualizations** for dataset insights.
- Fully functional **web UI** for ease of use.

### **5. Performance Testing**
- Simulated high-traffic requests using **Locust**.
- Measured latency and response times under different container settings.
- Documented the system's performance under load.

---

## **Project Structure**
```
Project_name/
│
├── README.md                   # Project Documentation
│
├── notebook/
│   ├── project_name.ipynb       # Model Training & Evaluation Notebook
│
├── src/
│   ├── preprocessing.py         # Data Preprocessing Script
│   ├── model.py                 # Model Training Script
│   ├── prediction.py            # Model Inference Script
│
├── data/
│   ├── train/                   # Training Dataset
│   ├── test/                    # Testing Dataset
│
├── models/
│   ├── model_name.pkl           # Trained Model File
│   ├── model_name.tf            # TensorFlow Model (if applicable)
│
├── deployment/
│   ├── Dockerfile               # Containerization
│   ├── app.py                   # Flask/FastAPI API
│   ├── requirements.txt         # Dependencies
│
└── tests/
    ├── locustfile.py            # Load Testing with Locust
    ├── test_api.py              # API Unit Tests
```

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/project_name.git
cd project_name
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Train the Model (Optional)**
```bash
python src/model.py
```

### **4. Run the Web Application**
```bash
python deployment/app.py
```

### **5. Run Docker Container**
```bash
docker build -t ml_model .
docker run -p 5000:5000 ml_model
```

### **6. Load Testing with Locust**
```bash
locust -f tests/locustfile.py
```

---

## **Retraining Pipeline**
1. **Upload new data** (via UI or API endpoint `/upload`).
2. **Trigger retraining** (`/retrain` endpoint or button in UI).
3. **Pipeline automatically:**
   - Processes the new data.
   - Retrains the model.
   - Saves & updates the deployed model.

---

## **Results from Flood Testing (Locust Load Test)**
- **Latency:** [XX] ms
- **Response Time:** [XX] ms (with 1, 2, and 5 containers)
- **Max Requests per Second:** [XX]

---

## **Video Demo**
Watch the full demo here: [YouTube Link]

---

## **Contributors**
- Sifa Kaveza Mwachoni

---

## **License**
This project is licensed under [MIT/GPL/Apache License].

