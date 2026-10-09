import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI


# ============================================================
# 1. APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Smart Delivery Time Predictor",
    page_icon="🚴",
    layout="centered"
)

st.title("🚴 Smart Delivery Time Predictor")
st.write(
    "Enter a delivery location, preparation time and rider code. "
    "The app will retrieve the other information automatically."
)

MODEL_PATH = "delivery_model.joblib"
RIDER_PATH = "Rider_Database_100.csv"
HOLIDAY_PATH = "Rajasthan_Holidays_Festivals_2026.csv"

INDIA_TZ = ZoneInfo("Asia/Kolkata")

# Identify this custom application when accessing public services.
APP_USER_AGENT = (
    "SmartDeliveryTimePredictor/1.0 "
    "(educational Streamlit application)"
)


# ============================================================
# 2. SECRETS AND MODEL
# ============================================================
def secret_value(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    pipeline = load_model()
except Exception as exc:
    st.error(
        "The trained model could not be loaded. Check that "
        "delivery_model.joblib exists and the scikit-learn "
        "version is compatible with the training environment."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# 3. LOAD CSV FILES; OPTIONAL SIDEBAR REPLACEMENTS
# ============================================================
st.sidebar.header("Data files")
st.sidebar.caption(
    "Upload updated CSVs if required. Otherwise, repository files "
    "will be used. Uploads are temporary to this running app session."
)

rider_upload = st.sidebar.file_uploader(
    "Rider database CSV",
    type=["csv"]
)

holiday_upload = st.sidebar.file_uploader(
    "Holiday calendar CSV",
    type=["csv"]
)


@st.cache_data
def load_default_csv(path):
    return pd.read_csv(path)


try:
    riders = (
        pd.read_csv(rider_upload)
        if rider_upload is not None
        else load_default_csv(RIDER_PATH)
    )

    holidays = (
        pd.read_csv(holiday_upload)
        if holiday_upload is not None
        else load_default_csv(HOLIDAY_PATH)
    )
except Exception as exc:
    st.error(
        "Could not read a CSV. Check the repository filenames "
        "or upload the correct files in the sidebar."
    )
    st.exception(exc)
    st.stop()


required_rider_columns = [
    "Rider_Code",
    "Experience_Years",
    "Vehicle_Type",
    "Status"
]

missing_columns = [
    col for col in required_rider_columns
    if col not in riders.columns
]

if missing_columns:
    st.error(
        "Rider CSV is missing columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if "Date" not in holidays.columns:
    st.error("Holiday CSV must contain a Date column.")
    st.stop()

riders["Rider_Code"] = (
    riders["Rider_Code"].astype(str).str.strip().str.upper()
)
riders["Status"] = (
    riders["Status"].astype(str).str.strip().str.lower()
)
riders["Vehicle_Type"] = (
    riders["Vehicle_Type"].astype(str).str.strip()
)

holidays["Date"] = pd.to_datetime(
    holidays["Date"], errors="coerce"
).dt.date


# ============================================================
# 4. ADDRESS GEOCODING WITH NOMINATIM
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(address):
    """
    Cached address search. No autocomplete.
    Respect the public Nominatim usage policy.
    """
    url = "https://nominatim.openstreetmap.org/search"

    response = requests.get(
        url,
        params={
            "q": address,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in"
        },
        headers={
            "User-Agent": APP_USER_AGENT
        },
        timeout=25
    )
    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError(
            "Address not found. Try adding the locality, city "
            "and PIN code."
        )

    item = results[0]

    return {
        "latitude": float(item["lat"]),
        "longitude": float(item["lon"]),
        "display_name": item["display_name"]
    }


# ============================================================
# 5. ROUTING WITH OSRM
# ============================================================
def get_route_details(origin_lat, origin_lon, dest_lat, dest_lon):
    """
    OSRM coordinates must be longitude,latitude.
    Public OSRM demo does not provide live traffic conditions.
    """
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    )

    response = requests.get(
        url,
        params={
            "overview": "false",
            "alternatives": "false"
        },
        headers={"User-Agent": APP_USER_AGENT},
        timeout=30
    )
    response.raise_for_status()

    result = response.json()

    if result.get("code") != "Ok" or not result.get("routes"):
        raise ValueError(
            "No driving route was returned for these coordinates."
        )

    route = result["routes"][0]

    return {
        "distance_km": route["distance"] / 1000,
        "route_duration_minutes": route["duration"] / 60
    }


# ============================================================
# 6. WEATHER WITH OPEN-METEO
# ============================================================
def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    response = requests.get(
        url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,precipitation,"
                "weather_code,cloud_cover"
            ),
            "timezone": "auto"
        },
        timeout=25
    )
    response.raise_for_status()

    data = response.json()
    current = data.get("current")

    if not current:
        raise ValueError("Weather data was not returned.")

    code = int(current.get("weather_code", 0))

    # These categories must match the model's training labels.
    if code in (0, 1):
        category = "Clear"
    elif code in (2, 3, 45, 48):
        category = "Cloudy"
    else:
        category = "Rain"

    return {
        "category": category,
        "temperature_c": current.get("temperature_2m"),
        "precipitation_mm": current.get("precipitation"),
        "cloud_cover_percent": current.get("cloud_cover"),
        "weather_code": code
    }


# ============================================================
# 7. CLOCK AND HOLIDAY CALENDAR
# ============================================================
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def as_yes_no(value):
    return (
        str(value).strip().lower()
        in {"1", "1.0", "true", "yes", "y"}
    )


def get_calendar_details(today):
    matching = holidays[holidays["Date"] == today]

    if matching.empty:
        return {
            "festival": "No",
            "event_name": "No listed event"
        }

    # If multiple events share a date, use the first flagged event.
    for _, row in matching.iterrows():
        flag = (
            as_yes_no(row["Festival_Flag"])
            if "Festival_Flag" in holidays.columns
            else True
        )

        if flag:
            name = str(
                row.get("Holiday_or_Festival", "Listed event")
            )
            return {
                "festival": "Yes",
                "event_name": name
            }

    return {
        "festival": "No",
        "event_name": "Calendar entry without Festival_Flag"
    }


# ============================================================
# 8. OPENAI EXPLANATION
# ============================================================
def generate_explanation(context):
    api_key = secret_value("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from Streamlit Secrets."
        )

    model_name = secret_value("OPENAI_MODEL", "gpt-4.1-mini")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        instructions=(
            "You explain delivery-time predictions to ordinary users. "
            "The supplied numerical prediction is authoritative. "
            "Never change it, recalculate it, or claim certainty. "
            "Do not claim causation or model accuracy without evidence. "
            "Clearly disclose assumptions and missing live traffic data."
        ),
        input=(
            "Explain this delivery prediction in plain language. "
            "Give a short summary and key details. Do not invent facts.\n\n"
            + json.dumps(context, indent=2, default=str)
        ),
        max_output_tokens=350
    )

    return response.output_text


# ============================================================
# 9. THREE-INPUT FORM
# ============================================================
st.subheader("New delivery")

with st.form("prediction_form"):
    destination = st.text_input(
        "Delivery location",
        placeholder="Locality, city and PIN code"
    )

    preparation_time = st.number_input(
        "Preparation time (minutes)",
        min_value=1,
        max_value=240,
        value=15,
        step=1
    )

    rider_code = st.text_input(
        "Rider code",
        placeholder="e.g. R001"
    )

    submitted = st.form_submit_button(
        "Fetch data and predict",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 10. FETCH DATA AND RUN MODEL
# ============================================================
if submitted:
    if not destination.strip() or not rider_code.strip():
        st.error("Please fill all three fields.")
        st.stop()

    rider_matches = riders[
        riders["Rider_Code"] == rider_code.strip().upper()
    ]

    if rider_matches.empty:
        st.error("Rider code was not found.")
        st.stop()

    rider = rider_matches.iloc[0]

    if rider["Status"] != "available":
        st.error("This rider is not marked as available.")
        st.stop()

    try:
        experience = float(rider["Experience_Years"])
        vehicle = str(rider["Vehicle_Type"]).strip()

        if pd.isna(experience) or not vehicle:
            raise ValueError(
                "The rider record has missing experience or vehicle data."
            )

        # The app assumes the configured origin is the restaurant.
        origin_lat = float(secret_value("RESTAURANT_LAT"))
        origin_lon = float(secret_value("RESTAURANT_LON"))

        now = datetime.now(INDIA_TZ)
        today = now.date()

        with st.spinner("Finding location, route and weather..."):
            destination_place = geocode_address(destination.strip())

            route = get_route_details(
                origin_lat,
                origin_lon,
                destination_place["latitude"],
                destination_place["longitude"]
            )

            weather = get_weather(
                destination_place["latitude"],
                destination_place["longitude"]
            )

        time_of_day = get_time_of_day(now.hour)
        weekend = "Yes" if now.weekday() >= 5 else "No"
        calendar = get_calendar_details(today)

        # The model was trained with ten features.
        # Order_Size is fixed to 1 because the form has only three inputs.
        # OSRM has no live traffic; Medium is a neutral placeholder only.
        model_features = pd.DataFrame([{
            "Distance_km": route["distance_km"],
            "Preparation_Time_min": float(preparation_time),
            "Order_Size": 1,
            "Courier_Experience_yrs": experience,
            "Weather": weather["category"],
            "Traffic_Level": "Medium",
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Weekend": weekend,
            "Festival": calendar["festival"]
        }])

        prediction = float(pipeline.predict(model_features)[0])
        prediction = max(0.0, prediction)

        context = {
            "predicted_delivery_minutes": round(prediction, 1),
            "delivery_location": destination_place["display_name"],
            "distance_km": round(route["distance_km"], 2),
            "OSRM_route_duration_minutes": round(
                route["route_duration_minutes"], 1
            ),
            "preparation_time_minutes": int(preparation_time),
            "rider_code": rider_code.strip().upper(),
            "rider_experience_years": experience,
            "vehicle_type": vehicle,
            "weather_category": weather["category"],
            "temperature_c": weather["temperature_c"],
            "precipitation_mm": weather["precipitation_mm"],
            "cloud_cover_percent": weather["cloud_cover_percent"],
            "weekday": now.strftime("%A"),
            "local_datetime": now.isoformat(),
            "time_of_day": time_of_day,
            "weekend": weekend,
            "festival": calendar["festival"],
            "calendar_event": calendar["event_name"],
            "traffic_limitation": (
                "Traffic_Level was set to Medium as a placeholder. "
                "OSRM does not provide live traffic."
            ),
            "order_size_assumption": 1
        }

        st.session_state["delivery_result"] = context
        st.session_state["model_features"] = model_features.to_dict(
            orient="records"
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Check your restaurant coordinates, "
            "CSV files and the external API responses."
        )
        st.exception(exc)
        st.stop()


# ============================================================
# 11. DISPLAY RESULT AND LLM EXPLANATION
# ============================================================
if "delivery_result" in st.session_state:
    result = st.session_state["delivery_result"]

    st.divider()
    st.subheader("Estimated delivery time")

    st.metric(
        "ML model estimate",
        f"{result['predicted_delivery_minutes']:.1f} minutes"
    )

    st.warning(
        "Prototype estimate based on a model trained with synthetic data. "
        "Actual delivery time may differ."
    )

    st.subheader("Automatically retrieved details")

    col1, col2 = st.columns(2)
    col1.metric("Road distance", f"{result['distance_km']:.2f} km")
    col2.metric("Weather", result["weather_category"])

    col3, col4 = st.columns(2)
    col3.metric("Rider experience", f"{result['rider_experience_years']} years")
    col4.metric("Traffic feature", "Medium (placeholder)")

    st.write("**Resolved destination:**", result["delivery_location"])
    st.write("**Local date and time:**", result["local_datetime"])
    st.write("**Day:**", result["weekday"])
    st.write("**Time of day:**", result["time_of_day"])
    st.write("**Calendar event:**", result["calendar_event"])
    st.write(
        f"**Weather details:** {result['temperature_c']} °C; "
        f"precipitation {result['precipitation_mm']} mm; "
        f"cloud cover {result['cloud_cover_percent']}%"
    )
    st.write(
        "**OSRM route duration (not live traffic):** "
        f"{result['OSRM_route_duration_minutes']:.1f} minutes"
    )

    st.caption(
        "Mapping attribution: © OpenStreetMap contributors. "
        "Routing: OSRM public demo server. Weather: Open-Meteo."
    )

    with st.expander("Inspect model input features"):
        st.json(st.session_state["model_features"])

    st.subheader("AI explanation")

    if st.button("Explain this prediction"):
        try:
            with st.spinner("Generating explanation with OpenAI..."):
                explanation = generate_explanation(result)
            st.markdown(explanation)
        except Exception as exc:
            st.error(
                "The ML prediction is available, but the AI explanation "
                "failed. Check the OpenAI API key, model access and billing."
            )
            st.exception(exc)
