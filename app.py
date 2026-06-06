
from flask import Flask, jsonify, request
from flask_cors import CORS

from predict import predict_gdm

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    required = [
        "Age", "Previous_Pregnancies", "Previous_C_Sections",
        "Diastolic_BP_mmHg", "Serum_Insulin_2h", "BMI"
    ]

    # Check all fields present
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predict_gdm(
            Age                  = float(data["Age"]),
            Previous_Pregnancies = float(data["Previous_Pregnancies"]),
            Previous_C_Sections  = float(data["Previous_C_Sections"]),
            Diastolic_BP_mmHg    = float(data["Diastolic_BP_mmHg"]),
            Serum_Insulin_2h     = float(data["Serum_Insulin_2h"]),
            BMI                  = float(data["BMI"]),
        )
        return jsonify({"input": data, "result": result})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)