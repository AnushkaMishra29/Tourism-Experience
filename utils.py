
from turtle import st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data (assuming the same directory structure)
def load_data():
    dataset=pd.read_csv("./Datasets/Final_FilteredDataset.csv")
    return dataset

def visit_data():
    visit_dataset=pd.read_excel('./Datasets/Mode.xlsx')
    return visit_dataset

def predict_rating(model, user_input):
    return model.predict(user_input)[0]

def predict_mode(model, user_input):
    return model.predict(user_input)[0]

def prepare_user_item_matrix(df):
    df_model_agg = df.groupby(['UserId', 'Attraction'])['Rating'].mean().reset_index()
    user_item_matrix = df_model_agg.pivot(index='UserId', columns='Attraction', values='Rating')
    return user_item_matrix.fillna(0)

def recommend_collaborative(user_id, user_item_matrix):
    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    if user_id not in user_item_matrix.index:
        return pd.Series(dtype=float)

    sim_scores = similarity_df[user_id].drop(user_id)
    sim_scores_aligned = sim_scores.reindex(user_item_matrix.index, fill_value=0)

    ratings = user_item_matrix
    weighted_ratings = ratings.T.dot(sim_scores_aligned)
    similarity_sum = sim_scores_aligned.sum()

    if similarity_sum == 0:
        return pd.Series(dtype=float)

    predicted_ratings = weighted_ratings / similarity_sum
    already_rated = ratings.loc[user_id][ratings.loc[user_id] > 0].index
    return predicted_ratings.drop(already_rated, errors='ignore').sort_values(ascending=False).head(5)

def recommend_content_based(user_id, user_item_matrix, attraction_data):
    attraction_data['features'] = (
        attraction_data['AttractionType'].fillna('') + ' ' +
        attraction_data['CityName'].fillna('') + ' ' +
        attraction_data['Region'].fillna('') + ' ' +
        attraction_data['Country'].fillna('') + ' ' +
        attraction_data['Continent'].fillna('')
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(attraction_data['features'])
    indices = pd.Series(attraction_data.index, index=attraction_data['Attraction'])

    liked_attractions = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] >= 4].index.tolist()
    if not liked_attractions:
        return attraction_data[['Attraction']].head(5)

    liked_indices = [indices[att] for att in liked_attractions if att in indices]
    sim_scores = linear_kernel(tfidf_matrix[liked_indices], tfidf_matrix).mean(axis=0)
    top_indices = sim_scores.argsort()[::-1]
    recommended_indices = [i for i in top_indices if attraction_data['Attraction'][i] not in liked_attractions]
    return pd.DataFrame({'Attraction': attraction_data['Attraction'].iloc[recommended_indices[:5]].values})

def hybrid_recommendation(user_id, user_item_matrix, attraction_data, alpha=0.55):
    collab_series = recommend_collaborative(user_id, user_item_matrix)
    content_df = recommend_content_based(user_id, user_item_matrix, attraction_data)

    collab_df = collab_series.reset_index()
    collab_df.columns = ['Attraction', 'PredictedRating_collab']
    collab_df['PredictedRating_collab'] = pd.to_numeric(collab_df['PredictedRating_collab'], errors='coerce').fillna(0)

    content_df = content_df.drop_duplicates()
    content_df['PredictedRating_content'] = 1.0

    merged = pd.merge(collab_df, content_df, on='Attraction', how='outer')
    merged['PredictedRating_collab'].fillna(0, inplace=True)
    merged['PredictedRating_content'].fillna(0, inplace=True)

    merged['HybridScore'] = alpha * merged['PredictedRating_collab'] + (1 - alpha) * merged['PredictedRating_content']
    return merged[['Attraction', 'HybridScore']].sort_values(by='HybridScore', ascending=False).reset_index(drop=True).head(10)

# Visualizations data
def plot_visualizations(df):
    top_attractions = df['AttractionName'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(y=top_attractions.index, x=top_attractions.values, ax=ax1)
    ax1.set_title("Top 10 Attractions")
    st.pyplot(fig1)

    top_regions = df['Region'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(y=top_regions.index, x=top_regions.values, ax=ax2)
    ax2.set_title("Top 10 Regions")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='VisitMode')
    ax3.set_title("Visit Mode Distribution")
    st.pyplot(fig3)
