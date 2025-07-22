

import streamlit as st
import joblib
import pandas as pd
import json
import os
from datetime import datetime
import googlemaps
from dotenv import load_dotenv
import numpy as np
import requests
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

import sys
def debug(msg):
    print(f"[DEBUG] {msg}", file=sys.stderr)
    sys.stderr.flush()

# --- Load Artifacts ---


@st.cache_resource(show_spinner=False)
def load_model():
    debug('Loading model...')
    return joblib.load('xgboost_model.pkl')

@st.cache_resource(show_spinner=False)
def load_imputer():
    debug('Loading imputer...')
    return joblib.load('imputer.pkl')

@st.cache_data(show_spinner=False)
def load_expected_features():
    debug('Loading expected_features...')
    with open('expected_features.json') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_reference_df():
    debug('Loading reference_df...')
    return pd.read_csv('reference_data.csv')

@st.cache_resource(show_spinner=False)
def get_gmaps_client():
    debug('Loading Google Maps API key...')
    load_dotenv()
    return googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))

model = load_model()
imputer = load_imputer()
expected_features = load_expected_features()
reference_df = load_reference_df()
gmaps = get_gmaps_client()
debug('Artifacts and gmaps client loaded.')

# --- Load Google Maps API Key ---

## (now handled by get_gmaps_client and above)

# --- Geocoding Function ---
def geocode_location(address):
    debug(f'Geocoding address: {address}')
    try:
        geocode_result = gmaps.geocode(address)
        debug(f'Geocode result: {geocode_result}')
        if geocode_result and len(geocode_result) > 0:
            lat = geocode_result[0]['geometry']['location']['lat']
            lon = geocode_result[0]['geometry']['location']['lng']
            debug(f'Geocoded lat/lon: {lat}, {lon}')
            return lat, lon
        else:
            debug('No geocode result found.')
            return None, None
    except Exception as e:
        debug(f'Geocode exception: {e}')
        return None, None

# --- Find Closest Location in Reference Data ---
def find_closest_location(lat, lon, df):
    debug(f'Finding closest location to lat={lat}, lon={lon}')
    distances = np.sqrt((df['Latitude'] - lat)**2 + (df['Longitude'] - lon)**2)
    idx = distances.idxmin()
    debug(f'Closest index: {idx}')
    return df.iloc[idx]

# --- Feature Extraction Function ---
def extract_features(closest_row, current_datetime, traffic_conditions, weather_conditions):
    debug(f'Extracting features for datetime={current_datetime}, traffic={traffic_conditions}, weather={weather_conditions}')
    hour = current_datetime.hour
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    day_of_week = current_datetime.weekday()
    rush_hour = int(7 <= hour <= 9 or 16 <= hour <= 19)
    night_time = int(hour < 6 or hour > 20)

    severity_encoded = closest_row['Severity_Encoded'] if 'Severity_Encoded' in closest_row else 0
    involved_encoded = closest_row['Involved_Encoded'] if 'Involved_Encoded' in closest_row else 0
    lanes_blocked_log = closest_row['Lanes_Blocked_Log'] if 'Lanes_Blocked_Log' in closest_row else 0
    advanced_risk = closest_row['Advanced_Risk_Score_Log'] if 'Advanced_Risk_Score_Log' in closest_row else 0
    high_incident_area = closest_row['High_Incident_Area'] if 'High_Incident_Area' in closest_row else 0

    # Encode weather_conditions to match training
    weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Storm': 3}
    if isinstance(weather_conditions, str):
        weather_encoded = weather_map.get(weather_conditions, 0)
    else:
        weather_encoded = 0
    features = {
        'sin_Hour': sin_hour,
        'cos_Hour': cos_hour,
        'DayOfWeek': day_of_week,
        'Rush_Hour': rush_hour,
        'Night_Time': night_time,
        'Severity_Encoded': severity_encoded,
        'Lanes_Blocked_Log': lanes_blocked_log,
        'Advanced_Risk_Score_Log': advanced_risk,
        'High_Incident_Area': high_incident_area,
        'Hour': hour,
        'Involved_Encoded': involved_encoded,
        'Traffic_Conditions': traffic_conditions,
        'Weather_Conditions': weather_encoded
    }
    debug(f'Extracted features: {features}')
    return pd.DataFrame([features])

def provide_context_and_suggestions(risk_score, traffic_conditions, weather_conditions, current_datetime):
    debug(f'Providing context for risk_score={risk_score}, traffic={traffic_conditions}, weather={weather_conditions}, datetime={current_datetime}')
    context = []
    suggestion = []

    if risk_score > 1.0:
        context.append("High risk due to historical data.")
        suggestion.append("Consider alternative routes or delay your travel.")
    elif risk_score > 0.8:
        context.append("Moderate risk due to historical data.")
        suggestion.append("Proceed with caution. Be prepared for potential delays.")
    elif risk_score > 0.5:
        context.append("Low risk due to historical data.")
        suggestion.append("Proceed with caution.")
    else:
        context.append("Very low risk due to historical data.")
        suggestion.append("You can proceed with your travel plans.")

    if traffic_conditions is not None:
        if traffic_conditions > 30:
            context.append(f"Heavy traffic with an estimated delay of {traffic_conditions:.1f} minutes.")
            suggestion.append("Consider leaving earlier or taking an alternative route.")
        elif traffic_conditions > 15:
            context.append(f"Moderate traffic with an estimated delay of {traffic_conditions:.1f} minutes.")
            suggestion.append("Be prepared for some delays.")
        else:
            context.append(f"Light traffic with an estimated delay of {traffic_conditions:.1f} minutes.")

    if weather_conditions != "Unknown":
        context.append(f"Current weather conditions: {weather_conditions}.")
        if weather_conditions in ["Rain", "Snow", "Storm"]:
            suggestion.append("Drive carefully due to adverse weather conditions.")

    if 7 <= current_datetime.hour <= 9 or 16 <= current_datetime.hour <= 19:
        context.append("Currently during rush hour.")
        suggestion.append("Expect higher traffic volumes.")
    if current_datetime.hour < 6 or current_datetime.hour > 20:
        context.append("Currently during night time.")
        suggestion.append("Be cautious of reduced visibility and potential fatigue.")

    context_str = " ".join(context)
    suggestion_str = " ".join(suggestion)
    return context_str, suggestion_str

# --- Helper: Get Real-Time Traffic (Google Maps Directions API) ---
def get_real_time_traffic(origin_lat, origin_lon, dest_lat, dest_lon):
    debug(f'Getting real-time traffic from ({origin_lat},{origin_lon}) to ({dest_lat},{dest_lon})')
    try:
        directions_result = gmaps.directions(
            (origin_lat, origin_lon),
            (dest_lat, dest_lon),
            mode="driving",
            departure_time="now"
        )
        debug(f'Directions result: {directions_result}')
        if directions_result and 'legs' in directions_result[0]:
            duration_in_traffic = directions_result[0]['legs'][0].get('duration_in_traffic')
            if duration_in_traffic:
                debug(f'Duration in traffic: {duration_in_traffic}')
                return duration_in_traffic['value'] / 60  # seconds to minutes
            else:
                debug('No duration_in_traffic, using duration')
                return directions_result[0]['legs'][0]['duration']['value'] / 60
        else:
            debug('No directions result legs found')
            return None
    except Exception as e:
        debug(f'Exception in get_real_time_traffic: {e}')
        return None
    
    # --- Helper: Get Real-Time Weather (OpenWeatherMap API) ---
def get_real_time_weather(lat, lon):
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    debug(f'Getting weather for lat={lat}, lon={lon}, url={url}')
    try:
        response = requests.get(url)
        debug(f'Weather API response: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            weather_main = data['weather'][0]['main']
            debug(f'Weather main: {weather_main}')
            return weather_main
        else:
            debug(f'Weather API non-200: {response.text}')
            return "Unknown"
    except Exception as e:
        debug(f'Exception in get_real_time_weather: {e}')
        return "Unknown"

# --- Streamlit App ---
st.title('Roadfie: Geospatial Temporal Risk Analysis (Web App)')
st.info('Note: This app uses the latest model and features. If you updated the .pkl/.json files, use "Clear Cache" in Streamlit if predictions do not change.')

current_location = st.text_input('Enter your current location')
destination = st.text_input('Enter your destination')




# Display date and time fields for user visibility (do not use for prediction)
now = datetime.now()
date_str = st.text_input('Date', value=now.strftime('%Y/%m/%d'))
time_str = st.text_input('Time', value=now.strftime('%H:%M'))

# Convert to datetime.date and datetime.time for prediction
try:
    date_time = datetime.strptime(date_str, '%Y/%m/%d').date()
    time = datetime.strptime(time_str, '%H:%M').time()
except Exception:
    date_time = now.date()
    time = now.time()




# --- Session State Logic ---
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
    st.session_state['context'] = None
    st.session_state['suggestion'] = None
    st.session_state['map'] = None
    st.session_state['risk_scores'] = None


if st.button('Predict Risk'):
    debug('Predict Risk button pressed.')
    current_lat, current_lon = geocode_location(current_location)
    dest_lat, dest_lon = geocode_location(destination)
    debug(f'Current lat/lon: {current_lat}, {current_lon}; Dest lat/lon: {dest_lat}, {dest_lon}')

    if current_lat is None or dest_lat is None:
        debug('Geocoding failed for one or both locations.')
        st.error('Could not geocode one or both locations. Please check your input.')
        st.session_state['prediction'] = None
        st.session_state['context'] = None
        st.session_state['suggestion'] = None
        st.session_state['map'] = None
        st.session_state['fig'] = None
    else:
        # --- Real-time traffic and weather ---
        traffic_conditions = get_real_time_traffic(current_lat, current_lon, dest_lat, dest_lon)
        debug(f'Traffic conditions: {traffic_conditions}')
        if traffic_conditions is None:
            debug('Traffic conditions None, using fallback.')
            traffic_conditions = 20  # fallback
        weather_conditions = get_real_time_weather(dest_lat, dest_lon)
        debug(f'Weather conditions: {weather_conditions}')

        closest_location = find_closest_location(dest_lat, dest_lon, reference_df)
        debug(f'Closest location row: {closest_location.to_dict()}')
        user_features = extract_features(closest_location, datetime.combine(date_time, time), traffic_conditions, weather_conditions)
        debug(f'User features before imputation: {user_features}')
        # Print extracted features for user visibility
        # Ensure all expected features are present in the DataFrame
        for col in expected_features:
            if col not in user_features.columns:
                user_features[col] = np.nan
        user_features = user_features[expected_features]
        st.write('Extracted Features for Prediction:')
        st.write(user_features)
        user_features_imputed = imputer.transform(user_features)
        user_features_df = pd.DataFrame(user_features_imputed, columns=expected_features)
        debug(f'User features after imputation: {user_features_df}')
        st.write('Imputed Features for Model:')
        st.write(user_features_df)

        predicted_score = model.predict(user_features_df)[0]
        debug(f'Predicted score: {predicted_score}')
        context, suggestion = provide_context_and_suggestions(predicted_score, traffic_conditions, weather_conditions, datetime.combine(date_time, time))
        debug(f'Context: {context} | Suggestion: {suggestion}')

        # --- Folium Map with Polyline for Optimized Route ---
        debug('Getting directions for folium map...')
        directions_result = gmaps.directions(
            (current_lat, current_lon),
            (dest_lat, dest_lon),
            mode="driving",
            departure_time="now"
        )
        debug(f'Directions result for map: {directions_result}')
        route_points = []
        if directions_result and 'legs' in directions_result[0]:
            steps = directions_result[0]['legs'][0]['steps']
            for step in steps:
                polyline = step['polyline']['points']
                decoded = googlemaps.convert.decode_polyline(polyline)
                for point in decoded:
                    route_points.append((point['lat'], point['lng']))
        debug(f'Route points: {route_points[:5]}... (total {len(route_points)})')
        m = folium.Map(location=[(current_lat + dest_lat) / 2, (current_lon + dest_lon) / 2], zoom_start=13)
        folium.Marker([current_lat, current_lon], tooltip='Start', icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([dest_lat, dest_lon], tooltip='Destination', icon=folium.Icon(color='red')).add_to(m)
        if route_points:
            folium.PolyLine(route_points, color='blue', weight=5, opacity=0.8).add_to(m)
        debug('Rendering folium map in Streamlit...')

        plot_df = reference_df.copy()
        debug(f'Plotting histogram with {len(plot_df)} rows.')
        # Use current context for all rows for demo purposes
        def make_features(row):
            return extract_features(row, datetime.combine(date_time, time), traffic_conditions, weather_conditions)
        features_list = [make_features(row) for _, row in plot_df.iterrows()]
        features_df = pd.concat(features_list, ignore_index=True)
        features_imputed = imputer.transform(features_df)
        features_imputed_df = pd.DataFrame(features_imputed, columns=expected_features)
        plot_df['Predicted_Risk_Score'] = model.predict(features_imputed_df)
        fig, ax = plt.subplots()
        ax.hist(plot_df['Predicted_Risk_Score'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Predicted Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predicted Risk Scores')
        ax.axvline(predicted_score, color='red', linestyle='dashed', linewidth=2, label=f'Predicted Score: {predicted_score:.2f}')
        ax.legend()
        debug('Rendering histogram in Streamlit...')

        # Store results in session state (store only data, not matplotlib figure)
        st.session_state['prediction'] = predicted_score
        st.session_state['context'] = context
        st.session_state['suggestion'] = suggestion
        st.session_state['map'] = m
        st.session_state['risk_scores'] = plot_df['Predicted_Risk_Score'].values


# --- Display Results if Available ---
if st.session_state.get('prediction') is not None:
    st.success(f"Predicted Risk Score: {st.session_state['prediction']:.2f}")
    st.info(f"Context: {st.session_state['context']}")
    st.warning(f"Suggestion: {st.session_state['suggestion']}")
    st_folium(st.session_state['map'], width=700, height=400)
    st.subheader('Distribution of Predicted Risk Scores')
    # Recreate the histogram plot on each rerun with updated context features
    plot_df = reference_df.copy()
    def make_features(row):
        return extract_features(row, datetime.combine(date_time, time), st.session_state.get('risk_scores_context_traffic', 20), st.session_state.get('risk_scores_context_weather', 'Clear'))
    features_list = [make_features(row) for _, row in plot_df.iterrows()]
    features_df = pd.concat(features_list, ignore_index=True)
    features_imputed = imputer.transform(features_df)
    features_imputed_df = pd.DataFrame(features_imputed, columns=expected_features)
    plot_df['Predicted_Risk_Score'] = model.predict(features_imputed_df)
    fig, ax = plt.subplots()
    risk_scores = plot_df['Predicted_Risk_Score']
    ax.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Predicted Risk Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Risk Scores')
    ax.axvline(st.session_state['prediction'], color='red', linestyle='dashed', linewidth=2, label=f'Predicted Score: {st.session_state["prediction"]:.2f}')
    ax.legend()
    st.pyplot(fig)

st.markdown('---')
st.markdown('**Note:** This app now uses real-time traffic and weather data. Ensure your API keys are set and you have internet connectivity.')