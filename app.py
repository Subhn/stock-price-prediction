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
if not os.path.exists('static'):
    os.makedirs('static')

# Load the model
try:
    model = load_model('stock_dl_model.h5')
    model.summary()  # Check the loaded model
except Exception as e:
    raise ValueError(f"Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get the current date in 'YYYY-MM-DD' format
    present_date = dt.datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')  # Default stock if none is entered
        
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()
        
        try:
            # Download stock data
            df = yf.download(stock, start=start, end=end)
            if df.empty:
                return render_template('index.html', error="No data found for the selected stock.", present_date=present_date)

            # Descriptive Data
            data_desc = df.head()
print("hello"
	    
            # Exponential Moving Averages
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()

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

            # Ensure correct input shape
            if x_test.shape[1:] != (100, 1):
                return render_template('index.html', error="Input data shape mismatch. Check the data preparation steps.", present_date=present_date)

            # Make predictions
            y_predicted = model.predict(x_test)

            # Inverse scaling for predictions
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df.Close, 'y', label='Closing Price')
            ax1.plot(ema20, 'g', label='EMA 20')
            ax1.plot(ema50, 'r', label='EMA 50')
            ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.legend()
            ema_chart_path = "static/ema_20_50.png"
            fig1.savefig(ema_chart_path)
            plt.close(fig1)

            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df.Close, 'y', label='Closing Price')
            ax2.plot(ema100, 'g', label='EMA 100')
            ax2.plot(ema200, 'r', label='EMA 200')
            ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Price")
            ax2.legend()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            fig2.savefig(ema_chart_path_100_200)
            plt.close(fig2)

            # Plot 3: Prediction vs Original Trend
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
            ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
            ax3.set_title("Prediction vs Original Trend")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Price")
            ax3.legend()
            prediction_chart_path = "static/stock_prediction.png"
            fig3.savefig(prediction_chart_path)
            plt.close(fig3)

            # Save dataset as CSV
            csv_file_path = f"static/{stock}_dataset.csv"
            df.to_csv(csv_file_path)

            # Return the rendered template with charts and dataset
            return render_template('index.html', 
                                   plot_path_ema_20_50=ema_chart_path, 
                                   plot_path_ema_100_200=ema_chart_path_100_200, 
                                   plot_path_prediction=prediction_chart_path, 
                                   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   dataset_link=csv_file_path, 
                                   present_date=present_date)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {e}", present_date=present_date)

    return render_template('index.html', present_date=present_date)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
