import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import time
import pytz

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Ensure the static folder exists for saving outputs
if not os.path.exists('static'):
    os.makedirs('static')

# Load the model
try:
    model = load_model('stock_dl_model.h5')
    model.summary()
except Exception as e:
    raise ValueError(f"Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    present_date = dt.datetime.now().strftime('%Y-%m-%d')

    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return render_template('index.html', error="No data found for the selected stock.", present_date=present_date)

            ticker_data = yf.Ticker(stock)
            max_price = ticker_data.history(period='max')['Close'].max()
            data_desc = df.describe()
# Format the max price for better readability
            max_price = f"{max_price:,.2f}" if isinstance(max_price, (int, float)) else 'N/A'


            # Exponential Moving Averages
            ema20 = df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            ema100 = df['Close'].ewm(span=100, adjust=False).mean()
            ema200 = df['Close'].ewm(span=200, adjust=False).mean()

            # Data splitting
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

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
                return render_template('index.html', error="Input data shape mismatch.", present_date=present_date)

            y_predicted = model.predict(x_test)
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            return render_template('index.html',
                                   stock_name=stock,
                                   #current_stock_price=current_price,
				   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   max_price=max_price,
                                   present_date=present_date)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}", present_date=present_date)

    return render_template('index.html', present_date=present_date)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
