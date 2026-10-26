
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("student_result_model.pkl")


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    missing = [key for key in features if key not in data]

    if missing:
        return jsonify({"missing_fields": missing}), 400

    try:
        values = [float(data[key]) for key in features]
        prediction = int(model.predict(np.array([values]))[0])
    except (ValueError, TypeError):
        return jsonify({"error": "Features must be numeric"}), 400

    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "prediction": result,
        "prediction_code": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
