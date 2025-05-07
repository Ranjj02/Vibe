# temp.py

import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API Configuration
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='3b6445d11279474baabf3cc18333e025',
    client_secret='726f6e72f659452c818cb09e8b7905d4'
))

# Load dataset and pipeline
spotify_data = pd.read_csv("data.csv")

with open("song_cluster_pipeline.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

# Feature columns
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# --- Helper Functions --- #

def find_song(name, artist):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} artist: {artist}', limit=1)

    if not results['tracks']['items']:
        return None

    track = results['tracks']['items'][0]
    track_id = track['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = name
    song_data['artist'] = artist
    song_data['explicit'] = int(track['explicit'])
    song_data['duration_ms'] = track['duration_ms']
    song_data['popularity'] = track['popularity']

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame([song_data])

def get_song_data(song, spotify_data):
    song_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower()) &
                             (spotify_data['artists'].str.contains(song['artist'], case=False, na=False))]
    if not song_data.empty:
        return song_data.iloc[0]
    else:
        return find_song(song['name'], song['artist'])

def recommend_songs(song, spotify_data, number_cols, cluster_pipeline, n_songs=10):
    song_df = get_song_data(song, spotify_data)
    
    if song_df is None or song_df.empty:
        st.warning("Could not find the song.")
        return []

    if isinstance(song_df, pd.Series):
        song_df = song_df.to_frame().T

    song_cluster = cluster_pipeline.predict(song_df[number_cols])[0]

    # Filter songs from the same cluster
    clustered_songs = spotify_data[spotify_data['cluster'] == song_cluster]

    # Cosine similarity
    scaler = cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(clustered_songs[number_cols])
    scaled_song_vector = scaler.transform(song_df[number_cols])

    distances = cdist(scaled_song_vector, scaled_data, metric='cosine')
    index = list(np.argsort(distances[0])[:n_songs])

    rec_songs = clustered_songs.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].str.lower().isin([song['name'].lower()])]

    return rec_songs[['name', 'year', 'artists']].to_dict(orient='records')

# --- Streamlit Frontend --- #

st.title("🎵 Vibe: Personalized Song Recommender")
st.subheader("Get music recommendations based on your favorite song!")

# Inputs
song_name = st.text_input("Enter a song name:")
artist_name = st.text_input("Enter the artist name:")

# Recommendation Button
if st.button("Get Recommendations"):
    if song_name and artist_name:
        recommendations = recommend_songs({'name': song_name, 'artist': artist_name},
                                          spotify_data, number_cols, cluster_pipeline)
        if recommendations:
            st.write("### Recommended Songs:")
            for rec in recommendations:
                st.write(f"**{rec['name']}** ({rec['year']}) by {rec['artists']}")
        else:
            st.warning("No recommendations found.")
    else:
        st.error("Please provide both song and artist name.")
