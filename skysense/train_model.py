"""
train_model.py
Run this script to retrain the Random Forest model on your own dataset.

Usage:
    python train_model.py --data flight_data.csv

Expected CSV columns:
    Airline, Source, Destination, Total_Stops,
    Dep_Hour, Dep_Minute, Arrival_Hour, Arrival_Minute,
    Duration_Hours, Duration_Minutes, Journey_Day, Journey_Month, Price
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

AIRLINES     = ['Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'GoAir',
                'Jet Airways', 'Air Asia', 'Multiple carriers']
SOURCES      = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai']
DESTINATIONS = ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad']

FEATURES = [
    'Airline_enc', 'Source_enc', 'Destination_enc', 'Total_Stops',
    'Dep_Hour', 'Dep_Minute', 'Arrival_Hour', 'Arrival_Minute',
    'Duration_Hours', 'Duration_Minutes', 'Journey_Day', 'Journey_Month',
]


def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic training data if no CSV is provided."""
    np.random.seed(42)
    airline_idx = np.random.randint(0, len(AIRLINES), n)
    source_idx  = np.random.randint(0, len(SOURCES), n)
    dest_idx    = np.random.randint(0, len(DESTINATIONS), n)
    stops       = np.random.choice([0, 1, 2, 3], n, p=[0.45, 0.35, 0.15, 0.05])
    dep_hour    = np.random.randint(0, 24, n)
    dep_min     = np.random.randint(0, 60, n)
    arr_hour    = np.random.randint(0, 24, n)
    arr_min     = np.random.randint(0, 60, n)
    dur_h       = np.random.randint(1, 10, n)
    dur_m       = np.random.randint(0, 60, n)
    j_day       = np.random.randint(1, 32, n)
    j_month     = np.random.randint(1, 13, n)

    ap = [1.25, 0.85, 0.80, 1.20, 0.78, 1.40, 0.75, 1.10]
    price = np.array([ap[a] for a in airline_idx]) * 4000
    price += stops * 600 + dur_h * 200
    price += ((dep_hour >= 6) & (dep_hour <= 9)) * 500
    price += ((dep_hour >= 17) & (dep_hour <= 20)) * 500
    price += np.random.normal(0, 600, n)
    price = np.clip(price, 1500, 25000).astype(int)

    return pd.DataFrame({
        'Airline': [AIRLINES[i] for i in airline_idx],
        'Source':  [SOURCES[i]  for i in source_idx],
        'Destination': [DESTINATIONS[i] for i in dest_idx],
        'Total_Stops': stops,
        'Dep_Hour': dep_hour, 'Dep_Minute': dep_min,
        'Arrival_Hour': arr_hour, 'Arrival_Minute': arr_min,
        'Duration_Hours': dur_h, 'Duration_Minutes': dur_m,
        'Journey_Day': j_day, 'Journey_Month': j_month,
        'Price': price,
    })


def train(df: pd.DataFrame):
    le_airline = LabelEncoder().fit(AIRLINES)
    le_source  = LabelEncoder().fit(SOURCES)
    le_dest    = LabelEncoder().fit(DESTINATIONS)

    df['Airline_enc']     = le_airline.transform(df['Airline'])
    df['Source_enc']      = le_source.transform(df['Source'])
    df['Destination_enc'] = le_dest.transform(df['Destination'])

    X = df[FEATURES]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R² Score : {r2_score(y_test, preds):.4f}")
    print(f"MAE      : ₹{mean_absolute_error(y_test, preds):,.0f}")

    joblib.dump(model, 'flight_rf.joblib')
    joblib.dump({'airline': le_airline, 'source': le_source, 'destination': le_dest}, 'encoders.joblib')
    print("Saved: flight_rf.joblib, encoders.joblib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to CSV dataset')
    args = parser.parse_args()

    if args.data:
        print(f"Loading dataset: {args.data}")
        df = pd.read_csv(args.data)
    else:
        print("No dataset provided — generating synthetic data …")
        df = generate_synthetic_data()

    train(df)
