from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("models/house_price_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            # Get form data
            size = float(request.form.get('size', 0))
            bedrooms = int(request.form.get('bedrooms', 0))
            bathrooms = int(request.form.get('bathrooms', 0))
            location = int(request.form.get('location', 0))

            # Prepare data and scale
            data = np.array([[size, bedrooms, bathrooms, location]])
            data_scaled = scaler.transform(data)

            # Make prediction
            prediction = round(model.predict(data_scaled)[0], 2)

        except ValueError:
            error_message = "Please enter valid numerical values."

    return render_template("index.html", prediction=prediction, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
