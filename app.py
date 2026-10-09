
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI


# =========================================================
# 1. APP CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Delivery Time Predictor",
    page_icon="🚴",
    layout="centered"
)

st.title("🚴 Smart Delivery Time Predictor")
st.write(
    "Enter the delivery location, preparation time and rider code. "
    "The app will retrieve the remaining information automatically."
)

MODEL_PATH = "delivery_model.joblib"
RIDER_FILE = "Rider_Database_100.csv"
HOLIDAY_FILE = "Rajasthan_Holidays_Festivals_2026.csv"

INDIA_TZ = ZoneInfo("Asia/Kolkata")

# Use an identifying User-Agent for OpenStreetMap's public service.
NOMINATIM_HEADERS = {
    "User-Agent": "SmartDeliveryTimeEducationalPrototype/1.0"
}


# =========================================================
# 2. LOAD THE TRAINED PIPELINE
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model_pipeline = load_model()
except Exception as exc:
    st.error(
        "The trained model could not be loaded. Confirm that "
        "delivery_model.joblib is in the GitHub repository and "
        "that scikit-learn is compatible with the training version."
    )
    st.code(str(exc))
    st.stop()


# =========================================================
# 3. LOAD CSV FILES
# Sidebar uploads override the repository's default CSV files.
# =========================================================
st.sidebar.header("Data management")
st.sidebar.caption(
    "Upload updated CSV files when needed. Otherwise, the app uses "
    "the files saved in GitHub."
)

rider_upload = st.sidebar.file_uploader(
    "Upload rider database CSV",
    type=["csv"],
    key="rider_upload"
)

holiday_upload = st.sidebar.file_uploader(
    "Upload holiday calendar CSV",
    type=["csv"],
    key="holiday_upload"
)


def load_csv(uploaded_file, default_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(default_path)


try:
    riders = load_csv(rider_upload, RIDER_FILE)
    holidays = load_csv(holiday_upload, HOLIDAY_FILE)
except Exception as exc:
    st.error(
        "Could not read a CSV. Check the default repository files "
        "or upload both CSVs in the sidebar."
    )
    st.code(str(exc))
    st.stop()


# Standardize rider codes and availability labels.
required_rider_columns = [
    "Rider_Code",
    "Experience_Years",
    "Vehicle_Type",
    "Status"
]

missing = [
    col for col in required_rider_columns
    if col not in riders.columns
]

if missing:
    st.error(
        "Rider CSV is missing columns: " + ", ".join(missing)
    )
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

if "Date" not in holidays.columns:
    st.error("Holiday CSV must contain a Date column.")
    st.stop()

holidays["Date"] = pd.to_datetime(
    holidays["Date"], errors="coerce"
).dt.date


# =========================================================
# 4. DATE, TIME AND HOLIDAY FEATURES
# =========================================================
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def is_true(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {
        "1", "1.0", "true", "yes", "y"
    }


def get_holiday(date_today):
    matches = holidays[holidays["Date"] == date_today]

    if matches.empty:
        return "No", "No matching calendar entry"

    row = matches.iloc[0]

    if "Festival_Flag" in holidays.columns:
        festival_flag = is_true(row["Festival_Flag"])
    else:
        # If no flag column exists, a matching event is treated
        # as a calendar event, not automatically as a festival.
        festival_flag = False

    event_name = str(
        row.get("Holiday_or_Festival", "Calendar event")
    ).strip()

    return (
        "Yes" if festival_flag else "No",
        event_name
    )


# =========================================================
# 5. ADDRESS -> COORDINATES USING NOMINATIM
# Public service: cache results and avoid autocomplete.
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_address(address):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": address,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "in"
        },
        headers=NOMINATIM_HEADERS,
        timeout=25
    )
    response.raise_for_status()

    results = response.json()
    if not results:
        raise ValueError(
            "Location not found. Enter a more specific address, "
            "including locality, city or PIN code."
        )

    place = results[0]

    return {
        "latitude": float(place["lat"]),
        "longitude": float(place["lon"]),
        "display_name": place.get("display_name", address)
    }


# =========================================================
# 6. WEATHER USING OPEN-METEO
# =========================================================
def get_weather(latitude, longitude):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,weather_code,cloud_cover"
            ),
            "timezone": "auto"
        },
        timeout=25
    )
    response.raise_for_status()

    data = response.json()
    current = data.get("current")

    if not current:
        raise ValueError("Weather API returned no current conditions.")

    code = int(current.get("weather_code", 0))

    # Simplify WMO weather codes into the model's training categories.
    if code in (0, 1):
        category = "Clear"
    elif code in (2, 3, 45, 48):
        category = "Cloudy"
    else:
        category = "Rain"

    return current, category


# =========================================================
# 7. ROAD DISTANCE AND ROUTE DURATION USING OSRM
# OSRM coordinate order is longitude,latitude.
# Public OSRM demo does NOT provide live traffic.
# =========================================================
def get_route(origin_lat, origin_lon, dest_lat, dest_lon):
    coordinates = (
        f"{origin_lon},{origin_lat};"
        f"{dest_lon},{dest_lat}"
    )

    url = (
        "https://router.project-osrm.org/"
        f"route/v1/driving/{coordinates}"
    )

    response = requests.get(
        url,
        params={
            "overview": "false",
            "alternatives": "false",
            "steps": "false"
        },
        headers=NOMINATIM_HEADERS,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(
            "OSRM could not find a driving route for these locations."
        )

    route = data["routes"][0]

    return {
        "distance_km": float(route["distance"]) / 1000,
        "duration_minutes": float(route["duration"]) / 60
    }


# =========================================================
# 8. OPENAI EXPLANATION
# OpenAI API usage may incur charges.
# =========================================================
def explain_prediction(context):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    model_name = st.secrets.get(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from Streamlit Secrets."
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        instructions=(
            "You explain a machine-learning delivery-time estimate. "
            "The numerical prediction is authoritative and must not "
            "be changed. Explain only the supplied data. Do not claim "
            "causation or model accuracy without evidence. Clearly "
            "mention that the traffic category is a placeholder, not "
            "live traffic, and that order size is fixed at 1. "
            "Use simple language and keep the explanation concise."
        ),
        input=(
            "Explain this delivery prediction for a non-technical user:\n"
            + json.dumps(context, indent=2, default=str)
        ),
        max_output_tokens=350
    )

    return response.output_text


# =========================================================
# 9. THREE-INPUT FORM
# =========================================================
st.subheader("New delivery")

with st.form("delivery_prediction_form"):
    destination = st.text_input(
        "1. Delivery location",
        placeholder="Locality, city, PIN code"
    )

    preparation_time = st.number_input(
        "2. Preparation time (minutes)",
        min_value=1,
        max_value=240,
        value=15,
        step=1
    )

    rider_code = st.text_input(
        "3. Rider code",
        placeholder="e.g. R001"
    )

    submitted = st.form_submit_button(
        "Fetch data and predict",
        type="primary",
        use_container_width=True
    )


# =========================================================
# 10. AUTOMATED DATA RETRIEVAL AND PREDICTION
# =========================================================
if submitted:
    if not destination.strip() or not rider_code.strip():
        st.error("Please fill in all three fields.")
        st.stop()

    now = datetime.now(INDIA_TZ)
    today = now.date()

    # Find rider.
    rider_matches = riders[
        riders["Rider_Code"] == rider_code.strip().upper()
    ]

    if rider_matches.empty:
        st.error("Rider code not found in the uploaded rider CSV.")
        st.stop()

    rider = rider_matches.iloc[0]

    if rider["Status"] != "available":
        st.error("This rider is not marked Available in the CSV.")
        st.stop()

    try:
        experience = float(rider["Experience_Years"])
        vehicle = str(rider["Vehicle_Type"]).strip()

        if pd.isna(experience) or not vehicle:
            raise ValueError(
                "Rider experience or vehicle information is missing."
            )

        # Restaurant origin coordinates must be configured in Secrets.
        origin_lat = float(st.secrets["RESTAURANT_LAT"])
        origin_lon = float(st.secrets["RESTAURANT_LON"])

        with st.spinner(
            "Finding location, route and current weather..."
        ):
            place = geocode_address(destination.strip())

            weather, weather_category = get_weather(
                place["latitude"], place["longitude"]
            )

            route = get_route(
                origin_lat,
                origin_lon,
                place["latitude"],
                place["longitude"]
            )

        weekend = "Yes" if now.weekday() >= 5 else "No"
        time_of_day = get_time_of_day(now.hour)
        festival, event_name = get_holiday(today)

        # NOTE:
        # Traffic_Level is set to Medium because public OSRM does not
        # provide live traffic. Order_Size is fixed at 1 because the
        # interface intentionally asks the user for only three inputs.
        model_input = pd.DataFrame([{
            "Distance_km": route["distance_km"],
            "Preparation_Time_min": float(preparation_time),
            "Order_Size": 1,
            "Courier_Experience_yrs": experience,
            "Weather": weather_category,
            "Traffic_Level": "Medium",
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Weekend": weekend,
            "Festival": festival
        }])

        # Predict using the saved preprocessing + regression pipeline.
        prediction = float(model_pipeline.predict(model_input)[0])
        prediction = max(0.0, prediction)

        result = {
            "predicted_delivery_minutes": round(prediction, 1),
            "destination_entered": destination.strip(),
            "resolved_destination": place["display_name"],
            "destination_latitude": place["latitude"],
            "destination_longitude": place["longitude"],
            "distance_km": round(route["distance_km"], 2),
            "route_duration_minutes": round(
                route["duration_minutes"], 1
            ),
            "preparation_time_minutes": int(preparation_time),
            "rider_code": rider_code.strip().upper(),
            "rider_experience_years": experience,
            "vehicle_type": vehicle,
            "weather_category": weather_category,
            "temperature_c": weather.get("temperature_2m"),
            "humidity_percent": weather.get("relative_humidity_2m"),
            "precipitation_mm": weather.get("precipitation"),
            "cloud_cover_percent": weather.get("cloud_cover"),
            "traffic_category": "Medium (placeholder; not live traffic)",
            "time_of_day": time_of_day,
            "weekday": now.strftime("%A"),
            "local_datetime": now.isoformat(),
            "weekend": weekend,
            "festival_flag": festival,
            "calendar_event": event_name,
            "order_size_assumption": 1
        }

        st.session_state["delivery_result"] = result

    except KeyError as exc:
        st.error(
            "A required Streamlit Secret is missing. Add RESTAURANT_LAT, "
            "RESTAURANT_LON and the OpenAI credentials as described "
            "in the deployment instructions."
        )
        st.code(str(exc))
        st.stop()

    except requests.RequestException as exc:
        st.error(
            "A public API request failed. Try again later or check "
            "whether the service is available."
        )
        st.code(str(exc))
        st.stop()

    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.code(str(exc))
        st.stop()


# =========================================================
# 11. SHOW RESULT + AI EXPLANATION
# =========================================================
if "delivery_result" in st.session_state:
    result = st.session_state["delivery_result"]

    st.divider()
    st.subheader("Delivery estimate")

    st.metric(
        "Predicted total delivery time",
        f"{result['predicted_delivery_minutes']:.1f} minutes"
    )

    st.caption(
        "This is an estimate from your trained model, not a guarantee. "
        "The training data is synthetic and needs real-world validation."
    )

    col1, col2 = st.columns(2)
    col1.metric("Road distance", f"{result['distance_km']:.2f} km")
    col2.metric("Weather", result["weather_category"])

    col3, col4 = st.columns(2)
    col3.metric("Rider experience", f"{result['rider_experience_years']} years")
    col4.metric("Route duration", f"{result['route_duration_minutes']:.1f} min")

    st.write("**Resolved location:**", result["resolved_destination"])
    st.write(
        f"**Current India time:** {result['local_datetime']} "
        f"({result['weekday']}, {result['time_of_day']})"
    )
    st.write("**Calendar entry:**", result["calendar_event"])
    st.write(
        f"**Current weather:** {result['temperature_c']} °C; "
        f"humidity {result['humidity_percent']}%; "
        f"precipitation {result['precipitation_mm']} mm; "
        f"cloud cover {result['cloud_cover_percent']}%."
    )
    st.write(
        "**Traffic feature:** Medium placeholder. This is not "
        "live traffic information."
    )

    with st.expander("Inspect the exact ML input features"):
        st.json({
            "Distance_km": result["distance_km"],
            "Preparation_Time_min": result["preparation_time_minutes"],
            "Order_Size": 1,
            "Courier_Experience_yrs": result["rider_experience_years"],
            "Weather": result["weather_category"],
            "Traffic_Level": "Medium",
            "Time_of_Day": result["time_of_day"],
            "Vehicle_Type": result["vehicle_type"],
            "Weekend": result["weekend"],
            "Festival": result["festival_flag"]
        })

    st.subheader("AI explanation")

    if st.button("Explain prediction with AI"):
        try:
            with st.spinner("Generating explanation..."):
                explanation = explain_prediction(result)
            st.markdown(explanation)
        except Exception as exc:
            st.error(
                "Prediction succeeded, but the AI explanation failed. "
                "Check your OpenAI API key, billing, model access and "
                "the app logs."
            )
            st.code(str(exc))

st.caption(
    "Mapping: © OpenStreetMap contributors | "
    "Weather: Open-Meteo"
)
