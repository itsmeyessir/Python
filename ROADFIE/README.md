# Roadfie: Geospatial Temporal Risk Analysis Web App

## Overview
Roadfie is a geospatial-temporal risk analysis tool for urban road safety, integrating real-time traffic and weather data with historical incident data. The app predicts risk scores for a given route and visualizes the results on an interactive map and histogram. This project is designed for demonstration at conferences and for public use.

## Main Files
- `roadfie_streamlit_app.py`: Main Streamlit web app for risk prediction and visualization.
- `expected_features.json`: List of features expected by the model for prediction.
- `reference_data.csv`: Historical incident data used for feature extraction and risk score distribution.
- `xgboost_model.pkl`: Trained XGBoost model for risk prediction.
- `imputer.pkl`: Preprocessing imputer for missing values.

## Features
- Predicts risk score for a user-specified route and time.
- Uses Google Maps API for geocoding and route/traffic data.
- Uses OpenWeatherMap API for real-time weather.
- Visualizes route and risk on an interactive map (Folium).
- Shows distribution of risk scores as a histogram.
- Provides context and suggestions based on prediction and real-time data.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd advml
```

### 2. Install Python and pip
- Python version: **3.9+** (recommended: 3.9 or 3.10)
- Pip version: **21.0+**

Check your versions:
```bash
python --version
pip --version
```

### 3. Install Dependencies
Install all required libraries using:
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file in the project root with the following:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```
- Get a Google Maps API key: https://developers.google.com/maps/documentation
- Get an OpenWeatherMap API key: https://openweathermap.org/api

### 5. Run the App
```bash
streamlit run roadfie_streamlit_app.py
```

The app will open in your browser. Enter your current location, destination, date, and time to get a risk prediction and visualizations.

## File Descriptions
- **roadfie_streamlit_app.py**: Main Streamlit app. Handles user input, geocoding, real-time API calls, feature extraction, prediction, and visualization.
- **expected_features.json**: JSON array of feature names required by the model.
- **reference_data.csv**: CSV of historical road incidents, used for feature engineering and plotting risk score distribution.
- **xgboost_model.pkl**: Trained XGBoost model (binary or regression) for risk prediction.
- **imputer.pkl**: Preprocessing imputer (e.g., SimpleImputer) for handling missing values in features.

## Requirements
See `requirements.txt` for all dependencies. Main libraries:
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- googlemaps
- python-dotenv
- requests
- folium
- streamlit-folium
- matplotlib

## Notes
- This app is for demonstration and research purposes. For production use, review API quotas, error handling, and security.
- Ensure your API keys are kept private and not committed to public repositories.
- For questions or issues, please contact the project maintainer or open an issue on GitHub.

---

## Example Usage
1. Enter your current location and destination (e.g., "UP Diliman" and "NU Manila").
2. The date and time of travel are automatically set.
3. Click "Predict Risk" to see the risk score, route, and risk distribution.

---
