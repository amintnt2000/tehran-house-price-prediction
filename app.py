from pathlib import Path
from flask import Flask, render_template, request
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
model = joblib.load(BASE_DIR / "model.joblib")
addresses = joblib.load(BASE_DIR / "addresses.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    error = None

    if request.method == "POST":
        try:
            data = pd.DataFrame([{
                "Area": float(request.form["area"]),
                "Room": int(request.form["room"]),
                "Parking": "parking" in request.form,
                "Warehouse": "warehouse" in request.form,
                "Elevator": "elevator" in request.form,
                "Address": request.form["address"],
            }])
            price = round(float(model.predict(data)[0]))
        except (ValueError, KeyError):
            error = "ورودی نامعتبر است."

    return render_template("index.html", addresses=addresses, price=price, error=error)


if __name__ == "__main__":
    app.run(debug=False)