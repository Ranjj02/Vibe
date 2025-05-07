
# main.py
# # Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist
import pickle
# Spotify and clustering-related imports
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configure Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='3b6445d11279474baabf3cc18333e025',
    client_secret='726f6e72f659452c818cb09e8b7905d4'
))

# Define helper functions

def find_song(name, artist):
    """Fetch song details from Spotify based on name and artist."""
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} artist: {artist}', limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['artist'] = [artist]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return song_data

def get_song_data(song, spotify_data):
    """Fetch song data from the dataset or Spotify."""
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                 & (spotify_data['artists'].str.contains(song['artist'], case=False))].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['artist'])

def get_mean_vector(song_list, spotify_data, number_cols):
    """Calculate the mean vector for a list of songs."""
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is not None:
            song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)
    return np.mean(song_vectors, axis=0) if song_vectors else None

def flatten_dict_list(dict_list):
    """Flatten a list of dictionaries into a single dictionary."""
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, spotify_data, number_cols, cluster_pipeline, n_songs=10):
    """Recommend songs based on input songs."""
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data, number_cols)
    if song_center is None:
        st.warning("Could not find any valid songs to base recommendations on.")
        return []

    scaler = cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    
    # Join the artists into a single string
    rec_songs['artists'] = rec_songs['artists'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return rec_songs[metadata_cols].to_dict(orient='records')

# Streamlit app
st.title("Vibe, Find songs you love🎵")
st.write("Get personalized song recommendations based on your favorite tracks!")

# Input fields for song name and artist name
song_name = st.text_input("Enter a song name:", "")
artist_name = st.text_input("Enter the artist's name:", "")

# Load pre-processed Spotify dataset and clustering pipeline (assumes these are preloaded)
spotify_data = pd.read_csv("data.csv")  # Replace with actual preloaded DataFrame
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Placeholder for your clustering pipeline

with open("song_cluster_pipeline.pkl", "rb") as f:
    cluster_pipeline = pickle.load(f)

def get_name(artist):
    text = ""
    for i in artist:
        if i not in ["[", "]", "\'"]:
            text += i
    return text
# Recommend songs when the user clicks the button
if st.button("Get Recommendations"):
    try:
        if song_name and artist_name:
            recommendations = recommend_songs([{'name': song_name, 'artist': artist_name}],
                                            spotify_data,
                                            number_cols,
                                            cluster_pipeline)
            if recommendations:
                st.write("### Recommended Songs:")
                for rec in recommendations:
                    st.write(f"**{rec['name']}** ({rec['year']}) by {get_name(rec['artists'])}")
            else:
                st.warning("No recommendations found. Please try another song.")
        else:
            st.error("Please enter both song name and artist name.")
    except:
        st.error("Could not find the song. Please try another song")
    
