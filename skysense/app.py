import logging
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import joblib
import os

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = joblib.load(os.path.join(BASE_DIR, "flight_rf.joblib"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.joblib"))

AIRLINES     = sorted(encoders["airline"].classes_.tolist())
SOURCES      = sorted(encoders["source"].classes_.tolist())
DESTINATIONS = sorted(encoders["destination"].classes_.tolist())

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "home.html",
        airlines=AIRLINES,
        sources=SOURCES,
        destinations=DESTINATIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Collect form data ────────────────────────────────────────────────
        airline     = request.form["airline"]
        source      = request.form["source"]
        destination = request.form["destination"]
        dep_time    = request.form["dep_time"]    # "HH:MM"
        arr_time    = request.form["arr_time"]    # "HH:MM"
        journey_date= request.form["journey_date"]# "YYYY-MM-DD"
        total_stops = int(request.form["total_stops"])

        logger.info(
            "Prediction request: %s → %s | %s | %s stops",
            source, destination, airline, total_stops,
        )

        # ── Validation ───────────────────────────────────────────────────────
        if source == destination:
            raise ValueError("Source and destination cannot be the same.")

        # ── Feature engineering ──────────────────────────────────────────────
        dep_h, dep_m   = map(int, dep_time.split(":"))
        arr_h, arr_m   = map(int, arr_time.split(":"))
        journey_dt     = pd.to_datetime(journey_date)
        journey_day    = journey_dt.day
        journey_month  = journey_dt.month

        # Duration (wrap midnight)
        dep_total  = dep_h * 60 + dep_m
        arr_total  = arr_h * 60 + arr_m
        dur_total  = arr_total - dep_total
        if dur_total <= 0:
            dur_total += 24 * 60          # next-day arrival
        dur_hours   = dur_total // 60
        dur_minutes = dur_total % 60

        # Encode categoricals
        airline_enc = int(encoders["airline"].transform([airline])[0])
        source_enc  = int(encoders["source"].transform([source])[0])
        dest_enc    = int(encoders["destination"].transform([destination])[0])

        # ── Prediction ───────────────────────────────────────────────────────
        features = [[
            airline_enc, source_enc, dest_enc, total_stops,
            dep_h, dep_m, arr_h, arr_m,
            dur_hours, dur_minutes,
            journey_day, journey_month,
        ]]
        feature_names = [
            "Airline_enc", "Source_enc", "Destination_enc", "Total_Stops",
            "Dep_Hour", "Dep_Minute", "Arrival_Hour", "Arrival_Minute",
            "Duration_Hours", "Duration_Minutes", "Journey_Day", "Journey_Month",
        ]
        df_input       = pd.DataFrame(features, columns=feature_names)
        predicted_price = int(model.predict(df_input)[0])

        # Feature importances for display
        importances    = model.feature_importances_
        fi_pairs       = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        top_features   = fi_pairs[:6]

        logger.info("Predicted price: ₹%d", predicted_price)

        return render_template(
            "results.html",
            price=f"{predicted_price:,}",
            airline=airline,
            source=source,
            destination=destination,
            total_stops=total_stops,
            dep_time=dep_time,
            arr_time=arr_time,
            journey_date=journey_dt.strftime("%d %b %Y"),
            duration=f"{dur_hours}h {dur_minutes}m",
            top_features=top_features,
        )

    except ValueError as ve:
        logger.warning("Validation error: %s", ve)
        return render_template(
            "home.html",
            error=str(ve),
            airlines=AIRLINES,
            sources=SOURCES,
            destinations=DESTINATIONS,
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        return render_template(
            "home.html",
            error="Something went wrong. Please try again.",
            airlines=AIRLINES,
            sources=SOURCES,
            destinations=DESTINATIONS,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
