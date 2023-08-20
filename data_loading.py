import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
from useful_functions import plot_spectrogram

from pandas import DataFrame as df
from sklearn.model_selection import train_test_split
from typing import Tuple
import pickle

# Loading saved items
with open('saved items/music_df.pkl', 'rb') as f:
    music_df = pickle.load(f)
with open('saved items/info_df_cleaned.pkl', 'rb') as f:
    info_df_cleaned = pickle.load(f)
with open('saved items/artists.pkl', 'rb') as f:
    artists = pickle.load(f)

# Goal: Create datasets and dataloaders
# for spectrograms (features) and three key metrics (labels): Energy, Danceability, and Valence.

# Audio characteristic names. I use "characteristics" instead of the term "features" used by Spotify to avoid confusion,
# since here they function as the "labels" (not features) in the machine learning context.
char_names = list(music_df.columns[8:18])
with open('saved items/char_names.pkl', 'wb') as f:
    pickle.dump(char_names, f)

# Make sure all label values are in the range [0,1]
def normalize(df, column):
    """Normalizes a column in a data frame to the range [0, 1]"""
    temp = (df.iloc[:, column] - min(df.iloc[:, column]))
    normalized = temp / max(temp)
    return normalized

music_df_normal = music_df
music_df_normal['Tempo'] = normalize(music_df, 9)
music_df_normal['Loudness'] = normalize(music_df, 17)

# Save music_df_normal
with open('saved items/music_df_normal.pkl', 'wb') as f:
    pickle.dump(music_df_normal, f)

# Split music_df_normal into train and test sets
TEST_SIZE = 0.2
df_train, df_test = train_test_split(music_df_normal, test_size=TEST_SIZE)
with open('saved items/df_test.pkl', 'wb') as f:
    pickle.dump(df_test, f)

# Custom PyTorch dataset class
class dfToDataset(Dataset):
    """
    Create torch.utils.data.Dataset from pandas data frame.
    If augment is True, transformations will be applied to features for data augmentation purposes.
    """
    def __init__(self,
                 feature_target_df: df,
                 feature_columns: list,
                 target_columns: list,
                 augment=False,
                 stretch_factor=0.8,
                 freq_mask=80,
                 time_mask=80):

        if augment:
            self.transform = torch.nn.Sequential(
                TimeStretch(stretch_factor, fixed_rate=True),
                FrequencyMasking(freq_mask_param=freq_mask),
                TimeMasking(time_mask_param=time_mask)
            )
        else:
            self.transform = None

        self.features = feature_target_df.iloc[:, feature_columns].values

        self.targets = feature_target_df.iloc[:, target_columns].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        feature = torch.tensor(self.features[:,0][idx], dtype=torch.float32).unsqueeze(dim=0)
        if self.transform:
            feature = self.transform(feature)
        feature = torch.nn.functional.normalize(feature)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return feature, target

    def target_names(self):
        return self.target_columns


# Create datasets: Energy, Danceability, and Valence

# Energy
energy_train_set = dfToDataset(feature_target_df=df_train,
                            feature_columns=[18],
                            target_columns=[11],
                            augment=True,
                            stretch_factor=0.5,
                            freq_mask=50,
                            time_mask=50)
energy_test_set = dfToDataset(feature_target_df=df_test,
                           feature_columns=[18],
                           target_columns=[11],
                           augment=False)

# Danceability (Dance)
dance_train_set = dfToDataset(feature_target_df=df_train,
                            feature_columns=[18],
                            target_columns=[10],
                            augment=True,
                            stretch_factor=0.5,
                            freq_mask=50,
                            time_mask=50)
dance_test_set = dfToDataset(feature_target_df=df_test,
                           feature_columns=[18],
                           target_columns=[10],
                           augment=False)

# Valence
valence_train_set = dfToDataset(feature_target_df=df_train,
                            feature_columns=[18],
                            target_columns=[13],
                            augment=True,
                            stretch_factor=0.5,
                            freq_mask=50,
                            time_mask=50)
valence_test_set = dfToDataset(feature_target_df=df_test,
                           feature_columns=[18],
                           target_columns=[13],
                           augment=False)

# Save datasets
with open('saved items/energy_train_set.pkl', 'wb') as f:
    pickle.dump(energy_train_set, f)
with open('saved items/energy_test_set.pkl', 'wb') as f:
    pickle.dump(energy_test_set, f)

with open('saved items/dance_train_set.pkl', 'wb') as f:
    pickle.dump(dance_train_set, f)
with open('saved items/dance_test_set.pkl', 'wb') as f:
    pickle.dump(dance_test_set, f)

with open('saved items/valence_train_set.pkl', 'wb') as f:
    pickle.dump(valence_train_set, f)
with open('saved items/valence_test_set.pkl', 'wb') as f:
    pickle.dump(valence_test_set, f)

# Visualize samples from train and test datasets
def visualize_dataset_features(train: torch.utils.data.Dataset,
                               test: torch.utils.data.Dataset):
    """Plots one spectrogram (feature) each from train and test datasets."""
    idtrain = random.randrange(len(train))
    idtest = random.randrange(len(test))
    plot_spectrogram(train[idtrain][0].squeeze(), title= "Train Sample")
    plot_spectrogram(test[idtest][0].squeeze(), title = "Test Sample")

# visualize_dataset_features(energy_train_set, energy_test_set)


# Create train and test data loaders
BATCH_SIZE = 8

energy_train_loader = DataLoader(dataset=energy_train_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
energy_test_loader = DataLoader(dataset=energy_test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

dance_train_loader = DataLoader(dataset=dance_train_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
dance_test_loader = DataLoader(dataset=dance_test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

valence_train_loader = DataLoader(dataset=valence_train_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
valence_test_loader = DataLoader(dataset=valence_test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

# Save dataloaders for training
with open('saved items/energy_train_loader.pkl', 'wb') as f:
    pickle.dump(energy_train_loader, f)
with open('saved items/energy_test_loader.pkl', 'wb') as f:
    pickle.dump(energy_test_loader, f)

with open('saved items/dance_train_loader.pkl', 'wb') as f:
    pickle.dump(dance_train_loader, f)
with open('saved items/dance_test_loader.pkl', 'wb') as f:
    pickle.dump(dance_test_loader, f)

with open('saved items/valence_train_loader.pkl', 'wb') as f:
    pickle.dump(valence_train_loader, f)
with open('saved items/valence_test_loader.pkl', 'wb') as f:
    pickle.dump(valence_test_loader, f)