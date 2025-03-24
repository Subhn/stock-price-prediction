import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Ensure the static folder exists for saving outputs
os.makedirs("static", exist_ok=True)

# Load the trained model
try:
    model = load_model("stock_dl_model.h5")
except Exception as e:
    raise ValueError(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    present_date = dt.datetime.now().strftime("%Y-%m-%d")

    if request.method == "POST":
        stock = request.form.get("stock", "POWERGRID.NS")
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return render_template("index.html", error="No data found for the selected stock.", present_date=present_date)

            ticker_data = yf.Ticker(stock)
            max_price = ticker_data.history(period="max")["Close"].max()
            data_desc = df.describe()

            # Format max price for better readability
            max_price = f"{max_price:,.2f}" if isinstance(max_price, (int, float)) else "N/A"

            # Data preprocessing
            data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            if x_test.shape[1:] != (100, 1):
                return render_template("index.html", error="Input data shape mismatch.", present_date=present_date)

            y_predicted = model.predict(x_test)
            scale_factor = 1 / scaler.scale_[0]
            y_predicted *= scale_factor
            y_test *= scale_factor

            return render_template(
                "index.html",
                stock_name=stock,
                data_desc=data_desc.to_html(classes="table table-bordered"),
                max_price=max_price,
                present_date=present_date,
            )

        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {e}", present_date=present_date)

    return render_template("index.html", present_date=present_date)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000, but let Render override
    app.run(host="0.0.0.0", port=port, debug=False)

