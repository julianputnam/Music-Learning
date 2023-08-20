# Music-Learning
## Machine learning regression project powered by the Spotify API

*Last updated 8.20.2023*

Given any 30 second music audio file, predict Energy, Danceability, and Valence (how upbeat the song sounds) using convolutional neural networks trained on data collected via the Spotify API for Python (spotipy). *Note: Only demo is currently available--ability to test on custom audio files and more features coming soon.*

Energy, Danceability, and Valence are audio characteristics stored by Spotify for every song on the platform and measured between 0 and 1. For instance, a song with Energy=0.2 and Valence=0.75 might be slow with soft instruments but still have a positive sentiment. Using our models, we can predict these values offline from previously "unheard" music audio files.

### Two ways to execute the code:
1. Use saved models and spectrograms to quickly see demo results: 
   Run model_demo.py

2. Collect data and structure data for PyTorch, train models, then view demo results:
   Run data_collection.py > data_loading.py > model_train.py > model_demo.py

### How:
• We collect and store information for ~3000 popular* songs in a pandas dataframe including Energy, etc., and download those songs' 30-second audio previews.
• We use torchaudio and a custom pipeline class to transform data from audio file > waveform (tensor) > spectrogram (tensor) for each collected audio file.
• We create multiple PyTorch datasets and dataloaders using audio spectrograms as features and each characteristic (Energy, Danceability, Valence) as a target.
• Train separate models, one for each target, on a custom CNN architecture. Learning spectrogram features is treated similarly to a computer vision problem.
• Demonstrate model capabilities using randomly selected song from test dataset ("unheard" during model training).

* Songs are pulled from the top songs of artists in top Spotify playlists and their related artists, so songs are not guaranteed to be popular or culturally relevant, though many are. This adds to the diversity of the collected data.

### Why?
I was motivated to start this project in June 2023 when I had the idea for an app that would allow a user to upload their own music audio and receive analysis and insight including the mood of the song, similar songs and artists, and possibly even which markets and demographics enjoy similar sounding songs the most, for indie artist marketing purposes. In order to bring this to fruition, the next steps would involve model implementation, accesible UI, and the building of additional features.
