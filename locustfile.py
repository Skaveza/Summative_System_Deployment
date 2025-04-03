from locust import HttpUser, task, between

class PredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def send_prediction_request(self):
        payload = {
            "services": "labor",
            "operating_hours": "9am-5pm",
            "care_system": "Public",
            "mode_of_payment": "Cash",
            "subcounty": "Abia Subcounty",
            "latitude": 1.2345,
            "longitude": 36.789
        }
        headers = {'Content-Type': 'application/json'}
        self.client.post("/predict/", json=payload, headers=headers)
