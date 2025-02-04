import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
MODEL_PATH = "model/stock_model.h5"  # Path to your trained model
SCALER_PATH = "model/scaler.pkl"  # Path to your saved scaler
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read dataset
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        if "Date" not in df.columns or "Close" not in df.columns:
            return "Invalid dataset! Ensure it contains 'Date' and 'Close' columns.", 400

        # Convert 'Date' to datetime format
        df["Date"] = pd.to_datetime(df["Date"])

        # Exponential Moving Averages
        ema20 = df["Close"].ewm(span=20, adjust=False).mean()
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df["Scaled_Close"] = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

        # Prepare data for model prediction
        x_test = []
        for i in range(100, len(df)):
            x_test.append(df["Scaled_Close"].values[i - 100 : i])  # Last 100 days

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM

        # Make predictions
        y_predicted = model.predict(x_test)

        # Inverse scale predictions
        y_predicted = scaler.inverse_transform(y_predicted)

        # Add predictions back to DataFrame
        df["Predicted"] = np.nan
        df.loc[len(df) - len(y_predicted) :, "Predicted"] = y_predicted.flatten()

        # Plot 1: Closing Price & EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["Date"], df["Close"], label="Closing Price", color="blue")
        ax1.plot(df["Date"], ema20, label="EMA 20", color="green")
        ax1.plot(df["Date"], ema50, label="EMA 50", color="red")
        ax1.set_title("Stock Price with EMA (20 & 50 Days)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        plot_ema = os.path.join(STATIC_FOLDER, "ema_chart.png")
        fig1.savefig(plot_ema)
        plt.close(fig1)

        # Plot 2: Actual vs Predicted
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df["Date"], df["Close"], label="Actual Price", color="green")
        ax2.plot(df["Date"], df["Predicted"], label="Predicted Price", color="red")
        ax2.set_title("Actual vs Predicted Stock Price")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        plot_prediction = os.path.join(STATIC_FOLDER, "prediction_chart.png")
        fig2.savefig(plot_prediction)
        plt.close(fig2)

        # Save processed dataset
        processed_csv = os.path.join(STATIC_FOLDER, f"processed_{file.filename}")
        df.to_csv(processed_csv, index=False)

        return render_template(
            "index.html",
            plot_ema=plot_ema,
            plot_prediction=plot_prediction,
            dataset_link=processed_csv,
        )

    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
