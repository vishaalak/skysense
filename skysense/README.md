# ✈ SKYSENSE – Flight Price Prediction System

A web-based flight fare prediction application built with **Flask** and a **Random Forest Regression** model.

> B.Sc. Computer Science (AI & DS) Project — VET Institute of Arts and Science, Erode  
> Submitted by: VISHAAL A K (Reg No: 2228K0116)  
> Guide: Mr. M. CHANDRU, Assistant Professor

---

## Features

- Predict flight prices based on airline, route, stops, timings and journey date
- Random Forest ML model (R² score: ~0.96)
- Clean, responsive web UI
- Feature importance visualization on results page
- Input validation & error handling with logging

---

## Project Structure

```
skysense/
├── app.py                  # Main Flask application
├── flight_rf.joblib        # Trained Random Forest model
├── encoders.joblib         # Label encoders for categorical features
├── requirements.txt        # Python dependencies
├── train_model.py          # Script to retrain the model
└── templates/
    ├── home.html           # Input form page
    └── results.html        # Prediction results page
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/skysense.git
cd skysense
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Input Parameters

| Field | Description |
|---|---|
| Source | Departure city |
| Destination | Arrival city |
| Date of Journey | Travel date |
| Departure Time | HH:MM |
| Arrival Time | HH:MM |
| Airline | Carrier name |
| Total Stops | 0 (non-stop) to 3+ |

---

## Tech Stack

- **Backend:** Python 3.x, Flask, Flask-CORS
- **ML Model:** scikit-learn RandomForestRegressor
- **Data Processing:** pandas, numpy, joblib
- **Frontend:** HTML5, CSS3 (no external frameworks)
- **Production:** Gunicorn + Nginx (recommended)

---

## Production Deployment (optional)

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## License

Academic project — VET Institute of Arts and Science, Erode (Affiliated to Bharathiar University, Coimbatore)
