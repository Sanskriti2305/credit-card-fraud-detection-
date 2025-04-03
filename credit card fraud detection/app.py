from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = "fraud_detection_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debugging: Print raw data received
        print("Raw request data:", request.get_data(as_text=True))
        print("JSON data:", request.json)  # Use request.json to get JSON data

        # Define the expected feature names
        feature_names = ['V14', 'V4', 'V3', 'V10', 'V12', 'V19', 'Amount', 'V8', 'V20', 'V6']

        # Extract input features from the received JSON data
        features = request.json.get("input_values", [])

        if len(features) != 10:
            raise ValueError("Expected 10 features, but got {}".format(len(features)))

        # Convert into a DataFrame
        input_data = pd.DataFrame([dict(zip(feature_names, features))])

        # Make prediction
        prediction = model.predict(input_data)
        fraud_status = "Fraud Detected!" if prediction[0] == 1 else "Legitimate Transaction"

        # Get fraud probability
        prob = model.predict_proba(input_data)
        fraud_probability = f"Fraud Probability: {prob[0][1]:.4f}"

        return jsonify({"prediction_text": fraud_status, "probability_text": fraud_probability})

    except Exception as e:
        print("Error:", str(e))  # Print error for debugging
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change port if needed



