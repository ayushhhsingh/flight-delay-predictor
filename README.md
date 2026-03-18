# ✈️ Flight Delay Predictor

A machine learning project that predicts whether a flight will be delayed based on various flight and airline features. The model is pre-trained and serialized for easy deployment.

---

## 📁 Project Structure

```
flight-delay-predictor/
│
├── flight_delays/          # Dataset or data processing scripts
├── delay_model.pkl         # Trained ML model (serialized)
├── label_encoders.pkl      # Label encoders for categorical features
├── scaler.pkl              # Feature scaler for numerical normalization
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed along with the following libraries:

```bash
pip install scikit-learn pandas numpy joblib
```

### Clone the Repository

```bash
git clone https://github.com/ayushhhsingh/flight-delay-predictor.git
cd flight-delay-predictor
```

---

## 🧠 How It Works

The predictor uses a pre-trained machine learning model to classify whether a flight will be delayed or not. The pipeline consists of three main components:

1. **`label_encoders.pkl`** — Encodes categorical features (e.g., airline name, origin/destination airports) into numeric values.
2. **`scaler.pkl`** — Normalizes numerical features (e.g., scheduled departure time, distance) to improve model performance.
3. **`delay_model.pkl`** — The core trained classifier that predicts flight delay status.

---

## 🔮 Making Predictions

Here's a sample code snippet to load the model and make predictions:

```python
import joblib
import pandas as pd

# Load model components
model = joblib.load('delay_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Example input (adjust features as per your dataset)
input_data = {
    'airline': ['AA'],
    'origin': ['JFK'],
    'destination': ['LAX'],
    'scheduled_departure': [1400],
    'distance': [2475]
}

df = pd.DataFrame(input_data)

# Encode categorical columns
for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

# Scale numerical features
df_scaled = scaler.transform(df)

# Predict
prediction = model.predict(df_scaled)
print("Delayed" if prediction[0] == 1 else "On Time")
```

---

## 📊 Dataset

The model was trained on the `flight_delays` dataset, which includes historical flight records with features such as:

- Airline carrier
- Origin and destination airports
- Scheduled departure and arrival times
- Flight distance
- Delay status (target variable)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| scikit-learn | Model training and preprocessing |
| pandas | Data manipulation |
| NumPy | Numerical computations |
| joblib | Model serialization |

---

## 📈 Model Details

- **Type:** Classification (Binary — Delayed / On Time)
- **Preprocessing:** Label Encoding + Standard Scaling
- **Output:** `1` → Delayed, `0` → On Time

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Ayush Singh**
- GitHub: [@ayushhhsingh](https://github.com/ayushhhsingh)
