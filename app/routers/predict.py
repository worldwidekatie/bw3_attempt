import logging
import random

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field, validator

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import json

import base64
import requests
import datetime
from urllib.parse import urlencode
import os
from dotenv import load_dotenv
from joblib import load


load_dotenv()

client_id=os.getenv('SPOTIPY_CLIENT_ID', default='OOPS')
client_secret=os.getenv('SPOTIPY_CLIENT_SECRET', default='OOPS')

model=load('knn_final.joblib')
df1 = pd.read_csv("https://raw.githubusercontent.com/BW-pilot/MachineLearning/master/CSVs/spotify_final.csv")
df = df1.drop(columns = ['artist_name', 'track_name'])
spotify = df.drop(columns = ['track_id'])
scaler = StandardScaler()
spotify_scaled = scaler.fit_transform(spotify)

log = logging.getLogger(__name__)
router = APIRouter()

def knn_predictor(audio_feats, k=20):
    """
    differences_df = knn_predictor(audio_features)
    """
    audio_feats_scaled = scaler.transform([audio_feats])

    ##Nearest Neighbors model
    knn = model
    
    # make prediction 
    prediction = knn.kneighbors(audio_feats_scaled)

    # create an index for similar songs
    similar_songs_index = prediction[1][0][:k].tolist()

    # Create an empty list to store simlar song names
    similar_song_ids = []
    similar_song_names = []

    # loop over the indexes and append song names to empty list above
    for i in similar_songs_index:
        song_id = df['track_id'].iloc[i]
        similar_song_ids.append(song_id)

    #################################################

    column_names = spotify.columns.tolist()

    # put scaled audio features into a dataframe
    audio_feats_scaled_df = pd.DataFrame(audio_feats_scaled, columns=column_names)

    # create empty list of similar songs' features
    similar_songs_features = []

    # loop through the indexes of similar songs to get audio features for each
    #. similar song
    for index in similar_songs_index:
        list_of_feats = spotify.iloc[index].tolist()
        similar_songs_features.append(list_of_feats)

    # scale the features and turn them into a dataframe
    similar_feats_scaled = scaler.transform(similar_songs_features)
    similar_feats_scaled_df = pd.DataFrame(similar_feats_scaled, columns=column_names)

    # get the % difference between the outputs and input songs
    col_names = similar_feats_scaled_df.columns.to_list()
    diff_df = pd.DataFrame(columns=col_names)
    for i in range(k):
        diff = abs(similar_feats_scaled_df.iloc[i] - audio_feats_scaled_df.iloc[0])
        diff_df.loc[i] = diff

    # add sums of differences 
    diff_df['sum'] = diff_df.sum(axis=1)
    diff_df = diff_df.sort_values(by=['sum'])
    diff_df = diff_df.reset_index(drop=True)

    # add track_id to DF
    diff_df['track_id'] = similar_song_ids

    # reorder cols to have track_id as first column
    cols = list(diff_df)
    cols.insert(0, cols.pop(cols.index('track_id')))
    diff_df = diff_df.loc[:, cols]

    # Remove the suggestion of the same song (all zeros)
    diff_df = diff_df[~(diff_df == 0).any(axis=1)]

    # Grab only the unique 10 songs
    diff_df = diff_df.drop_duplicates(subset=['track_id'])[:10]

    return diff_df



class Item(BaseModel):
    """Use this data model to parse the request body JSON."""
    trackid: str = Field(..., example='06AKEBrKUckW0KREUWRnvT')
    # acousticness: float = Field(..., example=0.5)
    # danceability: float = Field(..., example=0.7)
    # energy: float = Field(..., example=0.5)
    # liveness: float = Field(..., example=0.1)
    # loudness: float = Field(..., example=-11.8)
    # tempo: float = Field(..., example=-98.2)
    # valence: float = Field(..., example=-0.6)
    # instrumentalness: float = Field(..., example=-0.9)
    

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        dataframe = pd.DataFrame([dict(self)])
        return dataframe

class Searchitem(BaseModel):
    """Use this data model to parse the request body JSON."""
    user_query: str = Field(..., example='waterfalls')
    
    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        dataframe = pd.DataFrame([dict(self)])
        return dataframe


class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
            # return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token() 
        return token
    
    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        return headers
        
        
    def get_resource(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
    
    def get_album(self, _id):
        return self.get_resource(_id, resource_type='albums')
    
    def get_artist(self, _id):
        return self.get_resource(_id, resource_type='artists')
    
    def get_features(self, _id):
        return self.get_resource(_id, resource_type='audio-features')

    def get_track(self, _id):
        return self.get_resource(_id, resource_type='tracks')

    def search(self, query, search_type='artist' ): # type
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        data = urlencode({"q": query, "type": search_type.lower()})
        lookup_url = f"{endpoint}?{data}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()

spotipy = SpotifyAPI(client_id, client_secret)

@router.post('/predict')
async def predict(item: Item):    
    """Use a KNN model to made song predictions"""
    X_new = list(item)
    X_new = X_new[0][1]
    song_id = df[df['track_id']==X_new]
    song_id = song_id.drop(columns='track_id').values.tolist()
    diff_df = knn_predictor(song_id[0])
    something = diff_df.to_dict(orient='records')
    return JSONResponse(content=something)

@router.post('/search')
async def predict(searchitem: Searchitem):    
    """Does search by track"""
    X_new = list(searchitem)
    X_new = X_new[0][1]
    print(type(X_new))
    print(X_new)
    return spotipy.search(X_new, search_type="track")


@router.post('/track')
async def predict(item: Item):    
    """Does search for track by track id"""
    X_new = list(item)
    X_new = X_new[0][1]
    print(type(X_new))
    print(X_new)
    return spotipy.get_track(X_new)

@router.post('/artist')
async def predict(searchitem: Searchitem):    
    """Does search by artist"""
    X_new = list(searchitem)
    X_new = X_new[0][1]
    print(type(X_new))
    print(X_new)
    return spotipy.search(X_new, search_type="artist")

@router.post('/album')
async def predict(searchitem: Searchitem):    
    """Does search by album"""
    X_new = list(searchitem)
    X_new = X_new[0][1]
    print(type(X_new))
    print(X_new)
    return spotipy.search(X_new, search_type="album")

@router.post('/predict_all')
async def predict(item: Item):    
    """Use a KNN model to made song predictions"""
    X_new = list(item)
    X_new = X_new[0][1]
    features = spotipy.get_features(X_new)
    features = [features['acousticness'], features['danceability'],features['energy'], features['instrumentalness'], features['liveness'], features['loudness'], features['tempo'], features['valence']]
    diff_df = knn_predictor(features)
    something = diff_df.to_dict(orient='records')
    return JSONResponse(content=something)