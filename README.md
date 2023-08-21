# Music-Learning
## Machine learning regression project powered by the Spotify API

*Last updated 8.20.2023*

Given any 30 second music audio file, predict Energy, Danceability, and Valence (how upbeat the song sounds) using convolutional neural networks trained on data collected via Spotify API (spotipy). *Note: Only demo is currently available--ability to test on custom audio files and more features coming soon.*

Energy, Danceability, and Valence are audio characteristics stored by Spotify for every song on the platform and measured between 0 and 1. For instance, a song with Energy=0.2 and Valence=0.75 might be slow with soft instruments but still *feel* positive. Using trained models, we can predict these values offline from previously "unheard" music audio files.

Sample Spectrogram             |  Model Results
:-------------------------:|:-------------------------:
![Example of an audio spectrogram](https://i.ibb.co/HNqwGjJ/dont-be-shy-spec-demo.png)  |  ![Example of model results](https://i.ibb.co/HHFq0b9/dont-be-shy-demo.png)

### Two ways to execute the code:
1. Use saved models and spectrograms to quickly see demo results: 
   Run model_demo.py

2. Collect and structure data for PyTorch, train models, then view demo results:
   Run data_collection.py > data_loading.py > model_train.py > model_demo.py

Important: For the data collection steps, Spotify API access must be authorized with a valid client id and private key. To get your own, go to https://developer.spotify.com/

### How it's done:
• Collect and store information for ~3000 songs in a pandas dataframe including Energy, etc., and download those songs' 30-second audio previews.

• Use torchaudio and a custom pipeline class to transform data from audio file > waveform (tensor) > spectrogram (tensor) for each collected audio file.

• Create multiple PyTorch datasets and dataloaders using audio spectrograms as features and each characteristic (Energy, Danceability, Valence) as a target.

• Train separate models, one for each target, on a custom CNN architecture. Learning spectrogram features is treated similarly to a computer vision problem.

• Demonstrate model capabilities using a randomly selected song from test dataset ("unheard" during model training).

### Why?
I was motivated to start this project in June 2023 when I had the idea for an app that would allow a user to upload their own music audio and receive analysis and insight including the mood of the song, similar songs and artists, and possibly even which markets and demographics enjoy similar sounding songs the most, for indie artist marketing purposes. In order to bring this to fruition, the next steps would involve model implementation, accesible UI, building out additional features, and possibly redesigning the project based on an alternative data source containing marketing info, such as Chartmetric.
