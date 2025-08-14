from flask import Flask, render_template, request
import pickle, json, os
import pandas as pd
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# EXACT feature order from training
FEATURES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "region"]

# Load fitted scaler and fitted ridge model
with open("Standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("ridge.pkl", "rb") as f:
    ridge = pickle.load(f)

# Compose them into a single inference pipeline (no re-fitting happens)
pipe = Pipeline([("scaler", scaler), ("model", ridge)])

LAST_INPUTS_FILE = "last_inputs.json"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/input", methods=["GET", "POST"])
def input_page():
    # Prefill form with last inputs if available
    last_inputs = {}
    if os.path.exists(LAST_INPUTS_FILE):
        with open(LAST_INPUTS_FILE, "r") as f:
            last_inputs = json.load(f)

    if request.method == "POST":
        try:
            # Collect raw inputs as floats
            raw = {k: float(request.form[k]) for k in FEATURES}

            # Save latest inputs so you don’t have to retype
            with open(LAST_INPUTS_FILE, "w") as f:
                json.dump(raw, f)

            # Create a DataFrame with EXACT training columns
            X = pd.DataFrame([raw], columns=FEATURES)

            # Single call: scaler.transform -> ridge.predict (exactly like training)
            pred = pipe.predict(X)[0]

            return render_template("result.html", prediction=round(float(pred), 2), inputs=raw)
        except Exception as e:
            return f"Error: {e}"

    return render_template("input.html", last_inputs=last_inputs)
    
if __name__ == "__main__":
    app.run(debug=True)
