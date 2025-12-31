Hybrid Financial Market Prediction System

How to run:
1. Create a Python virtual environment (recommended):
   python -m venv venv
   venv\Scripts\activate   (Windows)
   source venv/bin/activate  (macOS / Linux)

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   python app.py

4. Open browser at: http://127.0.0.1:5000

Notes:
- TensorFlow may take longer to install and requires a compatible Python version.
- The sample data is in data/sample_stock_data.csv and contains synthetic stock prices.
- LSTM training (if TensorFlow available) runs for 10 epochs by default in this demo.