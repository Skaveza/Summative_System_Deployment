import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox
)

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict/"

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Healthcare Facility Prediction")
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()

        # Input fields
        self.services_input = QLineEdit(self)
        self.services_input.setPlaceholderText("Enter services (e.g., X-ray, Lab)")
        layout.addWidget(QLabel("Services:"))
        layout.addWidget(self.services_input)

        self.operating_hours_input = QLineEdit(self)
        self.operating_hours_input.setPlaceholderText("Operating hours (e.g., 9am-5pm)")
        layout.addWidget(QLabel("Operating Hours:"))
        layout.addWidget(self.operating_hours_input)

        self.care_system_input = QLineEdit(self)
        self.care_system_input.setPlaceholderText("Care system (Public/Private)")
        layout.addWidget(QLabel("Care System:"))
        layout.addWidget(self.care_system_input)

        self.payment_input = QLineEdit(self)
        self.payment_input.setPlaceholderText("Mode of Payment (e.g., Cash, Insurance)")
        layout.addWidget(QLabel("Mode of Payment:"))
        layout.addWidget(self.payment_input)

        self.subcounty_input = QLineEdit(self)
        self.subcounty_input.setPlaceholderText("Enter Subcounty")
        layout.addWidget(QLabel("Subcounty:"))
        layout.addWidget(self.subcounty_input)

        self.latitude_input = QLineEdit(self)
        self.latitude_input.setPlaceholderText("Latitude (e.g., 1.2345)")
        layout.addWidget(QLabel("Latitude:"))
        layout.addWidget(self.latitude_input)

        self.longitude_input = QLineEdit(self)
        self.longitude_input.setPlaceholderText("Longitude (e.g., 36.789)")
        layout.addWidget(QLabel("Longitude:"))
        layout.addWidget(self.longitude_input)

        # Predict button
        self.predict_button = QPushButton("Get Prediction", self)
        self.predict_button.clicked.connect(self.get_prediction)
        layout.addWidget(self.predict_button)

        # Output text area
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(QLabel("Prediction Output:"))
        layout.addWidget(self.output_text)

        self.setLayout(layout)

    def get_prediction(self):
        """ Sends input data to FastAPI and displays the response. """
        try:
            data = {
                "services": self.services_input.text(),
                "operating_hours": self.operating_hours_input.text(),
                "care_system": self.care_system_input.text(),
                "mode_of_payment": self.payment_input.text(),
                "subcounty": self.subcounty_input.text(),
                "latitude": float(self.latitude_input.text()),
                "longitude": float(self.longitude_input.text()),
            }

            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                prediction_result = response.json()
                self.output_text.setText(str(prediction_result))
            else:
                QMessageBox.warning(self, "Error", f"Failed to get prediction: {response.json()}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
