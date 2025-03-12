import numpy as np
import pandas as pd
from keras.models import load_model
from flask import Flask, render_template, request
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Ensure the static folder exists for saving outputs
if not os.path.exists('static'):
    os.makedirs('static')

# Load the model
try:
    model = load_model('stock_dl_model.h5')
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

            # Data splitting
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

            # Scaling data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Prepare data for prediction
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions
            y_predicted = model.predict(x_test)

            # Inverse scaling for predictions
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Extract required details
            current_price = y_predicted[-1][0]  # Latest predicted price
            max_price = max(y_predicted)[0]     # Maximum predicted price

            return render_template('index.html', 
                                   current_price=f"${current_price:.2f}",
                                   max_price=f"${max_price:.2f}",
                                   present_date=present_date)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}", present_date=present_date)

    return render_template('index.html', present_date=present_date)

if __name__ == '__main__':
    app.run(debug=True)
