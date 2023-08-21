import torch
import pickle
import random
from torch import nn
from pandas import DataFrame as df

# Goal: Demonstrate predicting characteristics from an unheard audio file; display results and percent accuracy.

# Load model class and saved models
class Spec_Analyzer(nn.Module):
    """Learns song characteristics from spectrograms.
    CNN regression model."""
    def __init__(self, input_shape: int, scans: int, output_shape: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=scans, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=scans, out_channels=scans, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=scans),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=scans, out_channels=scans, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=scans, out_channels=scans, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=scans),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=scans, out_channels=scans, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=scans, out_channels=scans, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=scans),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=scans*18*24, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv3(self.conv2(self.conv1(x))))

energy_model = torch.load("saved items/energy_model_0.pt")
dance_model = torch.load("saved items/dance_model_0.pt")
valence_model = torch.load("saved items/valence_model_0.pt")

# Load df_test to get test spectrograms and targets
with open('saved items/df_test.pkl', 'rb') as f:
    df_test = pickle.load(f)
df_test = df_test.reset_index()

def model_demo(energy_model: torch.nn.Module,
               dance_model: torch.nn.Module,
               valence_model: torch.nn.Module,
               df_test: df = df_test,
               device: str = 'cpu',
               seed: int = None):
    """
    Randomly choose a spectrogram from the test set.
    Print predicted Energy, Danceability, Valence and percent accuracy along with
    Artist_Name, Song_Name, and ground truth quantities.
    """
    random.seed(seed)
    rand_index = random.randrange(len(df_test))

    energy_model.to(device)
    dance_model.to(device)
    valence_model.to(device)

    test_spec = df_test['Spectrogram'].iloc[rand_index]
    test_spec_torch = torch.tensor(test_spec, dtype=torch.float32).unsqueeze(dim=0)
    test_spec_torch_normal = torch.nn.functional.normalize(test_spec_torch).unsqueeze(dim=0)

    with torch.inference_mode():
        energy_model.eval()
        dance_model.eval()
        valence_model.eval()
        pred_energy = energy_model(test_spec_torch_normal).item()
        pred_dance = dance_model(test_spec_torch_normal).item()
        pred_valence = valence_model(test_spec_torch_normal).item()

    original_data = df_test.loc[rand_index, ['Artist_Name', 'Song_Name', 'Energy', 'Danceability', 'Valence']]
    def get_accuracy():
        error = (
                        abs(original_data['Energy'] - pred_energy) / original_data['Energy'] +
                        abs(original_data['Danceability'] - pred_dance) / original_data['Energy'] +
                        abs(original_data['Valence'] - pred_valence) / original_data['Energy']
                ) / 3
        if 1 - error < 0:
            return "NA"
        return f"{(1-error) * 100:.3f}%"

    print(f"Index: {rand_index}\n"
          f"\nOriginal data:\n{original_data}\n"
          f"\nPredicted values:\n"
          f"{'Energy':<25} {pred_energy:^20.3f}\n"
          f"{'Danceability':<25} {pred_dance:^20.3f}\n"
          f"{'Valence':<25} {pred_valence:^20.3f}\n"
          f"\nAccuracy: {get_accuracy()}")

    return

model_demo(energy_model,
           dance_model,
           valence_model)
