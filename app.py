import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file, g
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

def get_model():
    """Load model only when needed to reduce memory usage."""
    if "model" not in g:
        model_path = "stock_dl_model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        try:
            g.model = load_model(model_path)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            g.model = None
    return g.model

@app.teardown_appcontext
def cleanup(exception=None):
    """Ensure model is removed from memory when not needed."""
    g.pop("model", None)

@app.route("/", methods=["GET", "POST"])
def index():
    present_date = dt.datetime.now().strftime("%Y-%m-%d")

    if request.method == "POST":
        stock = request.form.get("stock", "POWERGRID.NS").strip()
        start, end = dt.datetime.now() - dt.timedelta(days=5 * 365), dt.datetime.now()  # Last 5 years only

        try:
            # Download stock data
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return render_template("index.html", error="❌ No data found for the selected stock.", present_date=present_date)

            # Get stock metadata
            ticker_data = yf.Ticker(stock)
            max_price = ticker_data.history(period="max")["Close"].max()
            max_price = f"{max_price:,.2f}" if isinstance(max_price, (int, float)) else "N/A"

            # Preprocess data
            train_size = int(len(df) * 0.70)
            data_training = df["Close"][:train_size]
            data_testing = df["Close"][train_size:]

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

            past_50_days = data_training.tail(50)  # Reduce memory usage by using 50 days instead of 100
            final_df = pd.concat([past_50_days, data_testing])
            input_data = scaler.transform(final_df.values.reshape(-1, 1))

            x_test, y_test = [], []
            for i in range(50, len(input_data)):  # Adjusted for 50 days window
                x_test.append(input_data[i - 50:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            model = get_model()
            if model is None or x_test.shape[1:] != (50, 1):
                return render_template("index.html", error="❌ Model loading error or data shape mismatch.", present_date=present_date)

            y_predicted = model.predict(x_test)
            scale_factor = 1 / scaler.scale_[0]
            y_predicted *= scale_factor
            y_test *= scale_factor

            return render_template(
                "index.html",
                stock_name=stock,
                data_desc=df.describe().to_html(classes="table table-bordered"),
                max_price=max_price,
                present_date=present_date,
            )

        except Exception as e:
            print(f"❌ Error processing stock data: {e}")
            return render_template("index.html", error=f"❌ An error occurred: {e}", present_date=present_date)

    return render_template("index.html", present_date=present_date)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join("static", filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) else ("❌ File not found", 404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port binding
    app.run(host="0.0.0.0", port=port, debug=False)