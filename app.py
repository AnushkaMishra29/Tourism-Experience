
import streamlit as st
import pandas as pd
import joblib
from utils import load_data,visit_data, predict_rating, predict_mode, prepare_user_item_matrix, hybrid_recommendation, plot_visualizations

#raw data
df_raw = pd.read_csv('./Datasets/Final_Dataset.csv')
attraction_data = pd.read_csv('./Datasets/Attraction_Features.csv')
user_item_matrix = prepare_user_item_matrix(df_raw)

# Load models
rating_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")

# Load and preprocess data
df = load_data()
visit_df=visit_data()

# App layout
st.title("Tourism Experience Recommender")

user_id = st.selectbox("Select User ID", df['UserId'].unique())
print(user_id)
if user_id is None:
    st.error("Please select a valid User ID.")
    st.stop()
user_row = df[df['UserId'] == user_id].iloc[0]
print(user_row)
if user_row.empty:
    st.error("No data found for the selected User ID.")
    st.stop()
user_df = pd.DataFrame([user_row])
print(user_df.columns)


st.subheader("User Profile")
st.write(user_df[['Country', 'CityName', 'PreferredVisitMode', 'Rating']])

# Define regression features
regression_features = [
    'VisitYear', 'VisitMonth',
    'Continent_Africa', 'Continent_America', 'Continent_Asia',
    'Continent_Australia & Oceania', 'Continent_Europe',
    'Region', 'Country', 'CityName',
    'VisitMode_Couples', 'VisitMode_Family', 'VisitMode_Friends', 'VisitMode_Solo',
    'AvgRating', 'NumRatings'
]

# Define classification features
classification_features = [
    'VisitYear', 'VisitMonth', 'Rating', 'AttractionType', 'CityName', 'Country', 'Region',
    'Continent_Africa', 'Continent_America', 'Continent_Asia', 'Continent_Australia & Oceania', 'Continent_Europe',
    'VisitMode_Couples', 'VisitMode_Family', 'VisitMode_Friends', 'VisitMode_Solo',
    'AvgRating', 'NumRatings', 'PreferredVisitMode'
]

# --- Regression Feature Preparation ---
regression_input_df = pd.get_dummies(user_df[regression_features])
# Align with training features
trained_reg_features = rating_model.feature_names_in_
for col in trained_reg_features:
    if col not in regression_input_df.columns:
        regression_input_df[col] = 0
regression_input_df = regression_input_df[trained_reg_features]

predicted_rating = predict_rating(rating_model, regression_input_df)
st.subheader("Predicted Rating")
st.write(round(predicted_rating, 2))

# --- Classification Feature Preparation ---
class_input_df = pd.get_dummies(user_df[classification_features])
trained_class_features = classification_model.feature_names_in_
for col in trained_class_features:
    if col not in class_input_df.columns:
        class_input_df[col] = 0
class_input_df = class_input_df[trained_class_features]
# Get mapping from df
mode_mapping = visit_df.set_index('VisitModeId')['VisitMode'].to_dict()
predicted_mode = predict_mode(classification_model, class_input_df)
# Map to name
predicted_mode_name = mode_mapping.get(predicted_mode, "Unknown")

# Show in app
st.subheader("Predicted Visit Mode")
st.write(predicted_mode_name)


#Recommender System
if user_id in user_item_matrix.index:
    hybrid_recs = hybrid_recommendation(user_id, user_item_matrix, attraction_data, alpha=0.55)
    st.success("Top Recommendations Just for You 🎯")
    st.dataframe(hybrid_recs)
else:
    st.error(f"User ID {user_id} not found in the dataset.")


# Visualizations
st.subheader("Visualizations")
plot_visualizations(df)
