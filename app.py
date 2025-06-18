import streamlit as st
import pandas as pd
import joblib
from utils import (
    load_data, visit_data, predict_rating, predict_mode,
    prepare_user_item_matrix, hybrid_recommendation, plot_visualizations
)

# ------------------------
# Section 1: Load Data & Models
# ------------------------
st.set_page_config(page_title="Tourism Recommender", layout="wide")
st.title("🌍 Tourism Experience Recommender")

# Load datasets
df_raw = pd.read_csv('./Datasets/Final_Dataset.csv')
attraction_data = pd.read_csv('./Datasets/Attraction_Features.csv')
user_item_matrix = prepare_user_item_matrix(df_raw)

# Load models
rating_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")

# Preprocessed data
df = load_data()
visit_df = visit_data()

# ------------------------
# Section 2: User Input
# ------------------------
user_id = st.selectbox("Select User ID", df['UserId'].unique())

if user_id is None:
    st.error("Please select a valid User ID.")
    st.stop()

user_row = df[df['UserId'] == user_id]
if user_row.empty:
    st.error("No data found for the selected User ID.")
    st.stop()

user_df = user_row.reset_index(drop=True)

st.subheader("👤 User Profile")
st.write(user_df[['Country', 'CityName', 'PreferredVisitMode', 'Rating']])

# ------------------------
# Section 3: Predictions
# ------------------------

# --- Rating Prediction ---
regression_features = [
    'VisitYear', 'VisitMonth',
    'Continent_Africa', 'Continent_America', 'Continent_Asia',
    'Continent_Australia & Oceania', 'Continent_Europe',
    'Region', 'Country', 'CityName',
    'VisitMode_Couples', 'VisitMode_Family', 'VisitMode_Friends', 'VisitMode_Solo',
    'AvgRating', 'NumRatings'
]

reg_input = pd.get_dummies(user_df[regression_features])
for col in rating_model.feature_names_in_:
    if col not in reg_input.columns:
        reg_input[col] = 0
reg_input = reg_input[rating_model.feature_names_in_]

predicted_rating = predict_rating(rating_model, reg_input)

st.subheader("⭐ Predicted Rating")
st.write(round(predicted_rating, 2))

# --- Visit Mode Prediction ---
classification_features = [
    'VisitYear', 'VisitMonth', 'Rating', 'AttractionType', 'CityName', 'Country', 'Region',
    'Continent_Africa', 'Continent_America', 'Continent_Asia', 'Continent_Australia & Oceania', 'Continent_Europe',
    'VisitMode_Couples', 'VisitMode_Family', 'VisitMode_Friends', 'VisitMode_Solo',
    'AvgRating', 'NumRatings', 'PreferredVisitMode'
]

class_input = pd.get_dummies(user_df[classification_features])
for col in classification_model.feature_names_in_:
    if col not in class_input.columns:
        class_input[col] = 0
class_input = class_input[classification_model.feature_names_in_]

mode_mapping = visit_df.set_index('VisitModeId')['VisitMode'].to_dict()
predicted_mode_id = predict_mode(classification_model, class_input)
predicted_mode = mode_mapping.get(predicted_mode_id, "Unknown")

st.subheader("🧭 Predicted Visit Mode")
st.write(predicted_mode)

# ------------------------
# Section 4: Recommendations & Visuals
# ------------------------

# --- Recommender System ---
st.subheader("🎯 Personalized Recommendations")
if user_id in user_item_matrix.index:
    hybrid_recs = hybrid_recommendation(user_id, user_item_matrix, attraction_data, alpha=0.55)
    st.success("Top Recommendations Just for You")
    st.dataframe(hybrid_recs)
else:
    st.warning(f"No past data available for user {user_id} to generate recommendations.")


