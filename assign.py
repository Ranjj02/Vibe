# assign_clusters.py

import pandas as pd
import pickle

# Load dataset
spotify_data = pd.read_csv("data.csv")

# Load the trained cluster pipeline
with open("song_cluster_pipeline.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Assign clusters to the songs
spotify_data['cluster'] = cluster_pipeline.predict(spotify_data[number_cols])

# Save the updated dataset
spotify_data.to_csv("data.csv", index=False)
print("✅ Clusters successfully assigned and saved!")
