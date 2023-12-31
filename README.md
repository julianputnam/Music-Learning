# Music-Learning
## Machine learning regression project powered by the Spotify API

*Last updated 8.20.2023*

Given any 30 second music audio file, predict Energy, Danceability, and Valence (how upbeat the song sounds) using convolutional neural networks trained on data collected via Spotify API (spotipy). *Note: Only demo is currently available--ability to test on custom audio files and more features coming soon.*

Energy, Danceability, and Valence are audio characteristics stored by Spotify for every song on the platform, measured between 0 and 1. For instance, a song with Energy=0.2 and Valence=0.75 might be slow with soft instruments but still *feel* positive. Using trained models, we can predict these values offline from previously "unheard" music audio files.

Sample Spectrogram             |  Model Results
:-------------------------:|:-------------------------:
![Example of an audio spectrogram](https://i.ibb.co/HNqwGjJ/dont-be-shy-spec-demo.png)  |  ![Example of model results](https://i.ibb.co/HHFq0b9/dont-be-shy-demo.png)

### Two ways to execute the code:
1. Use saved models and spectrograms to quickly see demo results: 
   Run **model_demo.py**

2. Collect and structure data for PyTorch, train models, then view demo results:
   Run **data_collection.py** > **data_loading.py** > **model_train.py** > **model_demo.py**

Important: For the data collection steps, Spotify API access must be authorized with a valid client id and private key. To get your own, go to https://developer.spotify.com/

### How it's done
• Collect and store information for ~3000 songs in a pandas dataframe including Energy, etc., and download those songs' 30-second audio previews.

• Use torchaudio and a custom pipeline class to transform data from audio file > waveform (tensor) > spectrogram (tensor) for each collected audio file.

• Create multiple PyTorch datasets and dataloaders using audio spectrograms as features and each characteristic (Energy, Danceability, Valence) as a target. Augment training sets to help prevent overfitting. 

• Train separate models, one for each target, on a custom CNN architecture. Learning spectrogram features is treated similarly to a computer vision problem.

• Demonstrate models' predictive capabilities using a randomly selected song from test dataset ("unheard" during model training).

### Next steps:
• Improve existing code and build out additional features.

• Build model implementation and accesible UI.

• Explore alternative or additional data sources with richer datapoints.

• Seek viable use-cases and follow accordingly.
