import random

from config import cid, secret
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import torch, torchaudio
from torch import nn
from torchaudio.transforms import Resample, Spectrogram, MelScale
from torchvision import transforms as tvt

import os
import requests
from pathlib import Path
from pandas import DataFrame as df
import pickle

# Goal:
# Fill a pandas dataframe with a collection of about 3000 songs
# with columns including audio information and metrics from Spotify (e.g., Danceability).

# Authorize Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

# Initialize empty data frames
info_df = df(
    {
        # Artist and song essential data
        "Artist_Name": [], "Artist_ID": [], "Artist_Pop": [],
        "Song_Name": [], "Song_ID": [], "Song_Pop": [],
        "Preview_URL": [],

        # Song characteristics from Spotify
        "Key": [], "Mode": [],
        "Tempo": [],
        "Danceability": [], "Energy": [], "Liveness": [], "Valence": [],
        "Speechiness": [], "Acousticness": [], "Instrumentalness": [],
        "Loudness": []
    }
)

audio_df = df(
    {
        "Song_ID": [], "Spectrogram": []
    }
)

# First step of collection:
# Get dictionary of (Artist_Name, Artist_ID) (key, value)-pairs for prominent Spotify artists.
# Method: Get artists from two top playlists, then expand the network to those artists' "related artists"
# until the dictionary reaches a desired size (in the order of 10^3).

playlist_link = "https://open.spotify.com/playlist/37i9dQZEVXbNG2KDcFcKOF?si=1333723a6eff4b7f"
playlist_link_2 = "https://open.spotify.com/playlist/4hOKQuZbraPDIfaGbM3lKI"
playlist_URI = playlist_link.split("/")[-1].split("?")[0]
playlist_URI_2 = playlist_link_2.split("/")[-1].split("?")[0]

artists = {} # wil be dictionary of unique artists and IDs
for i, item in enumerate(sp.playlist_items(playlist_URI)["items"]):
    artist_id = item["track"]["artists"][0]["id"]
    artist_name = item["track"]["artists"][0]["name"]
    artists.update({f"{artist_name}": artist_id})
for i, item in enumerate(sp.playlist_items(playlist_URI_2)["items"]):
    artist_name = item["track"]["artists"][0]["name"]
    artist_id = item["track"]["artists"][0]["id"]
    artists.update({f"{artist_name}": artist_id})

# Expand artist and id dictionary using related artists request
moreartists = {}

for artist, id in artists.items():
    related = sp.artist_related_artists(id)
    for related_artist in related['artists']:
        artist_name = related_artist['name']
        artist_id = related_artist['id']
        moreartists.update({f"{artist_name}": artist_id})

# Merge dictionaries
artists.update(moreartists)

# Save artists dictionary
with open('saved items/artists.pkl', 'wb') as f:
    pickle.dump(artists, f)

# The artists dictionary will be used to download audio clips and populate the empty data frames.

def previews_available(artists_dict: dict, sample_size: int = None):
    """Takes dictionary of Spotify artists above and estimates how many artists have audio previews available.
    If a sample size smaller than the dictionary size is specified, it will estimate percentage based on a random sample.
    Recommended: Sample size of 50-100 for larger dictionaries."""

    def get_num_available(ids: list):
        num_available = 0
        for id in ids:
            preview_url = sp.artist_top_tracks(artist_id=id)['tracks'][0]['preview_url']
            if preview_url is None:
                continue
            else:
                num_available += 1
        return num_available

    if sample_size and sample_size < len(artists_dict):
        ids_to_check = random.sample(list(artists_dict.values()), sample_size)
        num_available = get_num_available(ids_to_check)
        print(f"Audio previews available in random sample: {num_available} of {sample_size}\n"
              f"Estimated percentage available overall: {round(num_available/sample_size*100, 2)}%")
    else:
        num_available = get_num_available(list(artists_dict.values()))
        print(f"Audio previews available: {num_available} of {len(artists_dict)}")

previews_available(artists_dict=artists, sample_size=20)


def populate_info_df(artists_dict: dict):
    """
    Key data collection step. Populates info_df with data requested from Spotify.
    Warning: Sends large number of API requests to Spotify. Number of requests >= 5 * len(artists_dict).
    Spotify begins to restrict API requests somewhere around 100k requests.
    """
    for artist, id in artists_dict.items():

        artist_request = sp.artist(id)
        tracks_request = sp.artist_top_tracks(artist_id=id)

        i = 0
        while i < 3 and i < len(tracks_request['tracks']):
            # Updates info_df for each track (up to three tracks) in artist's top tracks
            track = tracks_request['tracks'][i]

            Artist_Name = artist
            Artist_ID = id
            Artist_Pop = artist_request['popularity']
            Song_Name = track['name']
            Song_ID = track['id']
            Song_Pop = track['popularity']
            Preview_URL = track['preview_url']

            char_request = sp.audio_features(Song_ID)

            Key = char_request[0]['key']
            Mode = char_request[0]['mode']
            Tempo = char_request[0]['tempo']
            Danceability = char_request[0]['danceability']
            Energy = char_request[0]['energy']
            Liveness = char_request[0]['liveness']
            Valence = char_request[0]['valence']
            Speechiness = char_request[0]['speechiness']
            Acousticness = char_request[0]['acousticness']
            Instrumentalness = char_request[0]['instrumentalness']
            Loudness = char_request[0]['loudness']

            info_df.loc[len(info_df)] = [
                Artist_Name, Artist_ID, Artist_Pop,
                Song_Name, Song_ID, Song_Pop,
                Preview_URL,

                Key, Mode,
                Tempo,
                Danceability, Energy, Liveness, Valence,
                Speechiness, Acousticness, Instrumentalness,
                Loudness
            ]

            i += 1

# Testing populate_info_df():
# artists_shortened = dict(list(artists.items())[:20])
# populate_info_df(artists_shortened)
# info_df.head()

# Populate columns of info_df:
populate_info_df(artists)

# Drop duplicates and songs without a preview URL (URL will be necessary at the next step).
info_df_cleaned = info_df.drop_duplicates(subset='Song_ID').reset_index(drop=True)
info_df_cleaned = info_df_cleaned.dropna(subset=['Preview_URL']).reset_index(drop=True)

# Save info_df_cleaned
with open('saved items/info_df_cleaned.pkl', 'wb') as f:
    pickle.dump(info_df_cleaned, f)

# Next step: Download 30-second audio previews for each song in info_df_cleaned
def get_preview(folder: str,
                filename: str,
                preview_url: str):
    """Checks if an audio preview has been downloaded. If not, downloads audio file."""

    if Path(f"{folder}/{filename}").is_file():
        print("Already exists, skipping download...")
    else:
        print(f"Downloading {filename}")
        request = requests.get(preview_url)
        with open(f"{folder}/{filename}", 'wb') as f:
            f.write(request.content)

def download_previews_from_df(folder: str, df: df = info_df_cleaned):
    """Uses get_preview() function to download available audio previews from info_df_cleaned (or similar data frame)."""

    for i in range(len(df)):
        if df['Preview_URL'][i] is None:
            print(f"Sorry, no preview is available for {df['Song_Name'][i]} by {df['Artist_Name'][i]}")
            continue
        else:
            get_preview(folder=folder,
                        filename=f"{df['Song_ID'][i]}.mp3",
                        preview_url=df['Preview_URL'][i])

download_previews_from_df(folder="MAN_previews")

preview_files = os.listdir("MAN_previews/")
if ".DS_Store" in preview_files: preview_files.remove(".DS_Store")

# Generate waveforms and then spectrograms from downloaded audio files
def wf_sr_generator(folder: str, preview_files: list):
    """Generates waveform, sample rate, and song ID from local audio previews."""

    for i, file in enumerate(preview_files):
        wf, sr = torchaudio.load(f"{folder}/{preview_files[i]}")
        id = preview_files[i].split(sep=".")[0] #extracts song id from downloaded file name

        yield wf, sr, id

class AudioPipeline(nn.Module):
    """
    Custom pipeline class to convert from waveform tensor to spectrogram tensor.
    A spectrogram is a visual representation of audio as a 2D, 1-channel image.
    These audio spectrograms can then be used to train a CNN in the same way as regular images.
    """

    def __init__(self,
                 input_freq: int = 44100,
                 resample_freq: int = 8000,
                 n_fft=1024,
                 n_mel=256,
                 stretch_factor=0.8):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq,
                                 new_freq=resample_freq)
        self.spec = Spectrogram(n_fft=n_fft, power=2)

        self.mel_scale = MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        self.resize = tvt.Compose([
            tvt.Resize(size=(150,200))
        ])

    def forward(self, waveform: torch.Tensor, augment: bool = False) -> torch.Tensor:
        resampled = self.resample(waveform) #Changes sample rate (less data)
        spec = self.spec(resampled) #Spectrogram
        mel = self.mel_scale(spec) #Changes frequency from linear scale to more like how human ear hears it
        mono = (mel[0] + mel[1]) / 2
        unsqueezed = mono.unsqueeze(dim=0)
        with torch.inference_mode():
            resized = self.resize(unsqueezed)
        squeezed = resized.squeeze()
        final_spec = squeezed.detach().numpy()
        return final_spec

pipeline = AudioPipeline()

def populate_audio_df(df: df, folder: str, preview_files: list, augment:bool = True):
    """Populates audio_df (or similar dataframe) with song IDs and spectrograms."""

    for waveform, sample_rate, id in wf_sr_generator(folder, preview_files):

        spectrogram = pipeline(waveform, augment=False)

        df.loc[len(audio_df)] = [id, spectrogram] #populates audio_df

# Populate columns of audio_df
populate_audio_df(df=audio_df, folder="MAN_previews", preview_files=preview_files)

# Merge dataframes matching on Song_ID
music_df = info_df_cleaned.merge(right=audio_df, on='Song_ID')

# Save music_df
with open('saved items/music_df.pkl', 'wb') as f:
    pickle.dump(music_df, f)

# Visualize a random spectrogram from music_df:
# from useful_functions import random_spectrogram
# random_spectrogram(df=music_df)
