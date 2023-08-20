import torch
from torch.utils.data import DataLoader
from torch import nn

import pickle
import random
from tqdm.auto import tqdm
from timeit import default_timer as timer
from useful_functions import plot_loss_curves

# Goal: train CNN regression models to estimate characteristics (e.g., Energy, Valence) of an unheard audio file.

with open('saved items/char_names.pkl', 'rb') as f:
    char_names = pickle.load(f)

# Load dataloaders and label names
with open('saved items/energy_train_loader.pkl', 'rb') as f:
    energy_train_loader = pickle.load(f)
with open('saved items/energy_test_loader.pkl', 'rb') as f:
    energy_test_loader = pickle.load(f)

with open('saved items/dance_train_loader.pkl', 'rb') as f:
    dance_train_loader = pickle.load(f)
with open('saved items/dance_test_loader.pkl', 'rb') as f:
    dance_test_loader = pickle.load(f)

with open('saved items/valence_train_loader.pkl', 'rb') as f:
    valence_train_loader = pickle.load(f)
with open('saved items/valence_test_loader.pkl', 'rb') as f:
    valence_test_loader = pickle.load(f)


# Model architecture
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
        # If altering model, uncomment lines 68-71 and comment line 72 to test for input shape into classifier block
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # print(x.shape) # input shape for classifier block
        return self.classifier(self.conv3(self.conv2(self.conv1(x))))

# Test one input spectrogram (especially to get input shape for model's classifier block after altering model)
device = 'cpu'
test_model = Spec_Analyzer(input_shape=1, scans=10, output_shape=1).to(device)
random_index = random.randrange(8)
test_spec = next(iter(energy_train_loader))[0][random_index].unsqueeze(dim=0)
with torch.inference_mode():
    test_model.eval()
    test_model(test_spec)  # shape before classifier: [1, 10, 18, 24]

# Train and testing loop
def train_model(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device=device,
               seed=42,
               epochs: int=5):

    model.to(device)

    results={
        "Train Loss": [],
        "Test Loss":  []
    }

    torch.manual_seed(seed)

    start_time = timer()

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        test_loss = 0

        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                loss = loss_fn(y_logits, y)
                test_loss += loss

        train_loss = train_loss / len(train_dataloader)
        test_loss = test_loss / len(test_dataloader)
        results["Train Loss"].append(train_loss)
        results["Test Loss"].append(test_loss)

        print(f"\nEpoch: {epoch} | Train Loss: {train_loss} | Test Loss: {test_loss}")# | Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        scheduler.step() #adjust learning rate

    end_time = timer()
    print(f"Model trained in {end_time-start_time:.4f} seconds.")

    return results

# Plot model loss curves
def loss_curves(results: dict):
    train_loss_numpy = []
    test_loss_numpy = []
    for key, value in results.items():
        for tensor in value:
            if key == "Train Loss":
                train_loss_numpy.append(tensor.cpu().detach().numpy())
            elif key == "Test Loss":
                test_loss_numpy.append(tensor.cpu().numpy())

    plot_loss_curves(train_loss_numpy, test_loss_numpy)


# Train, test, and evaluation

device = 'mps' if getattr(torch, "has_mps") else 'cpu'  # change to available/desired processing unit

# Initialize models, loss_fn
energy_model_0 = Spec_Analyzer(input_shape=1, scans=20, output_shape=1).to(device)
dance_model_0 = Spec_Analyzer(input_shape=1, scans=20, output_shape=1).to(device)
valence_model_0 = Spec_Analyzer(input_shape=1, scans=20, output_shape=1).to(device)

loss_fn = nn.HuberLoss(delta=0.06)

# Train models
# Energy
optimizer = torch.optim.Adam(params=energy_model_0.parameters(), lr=0.01)
SCHED_TIMESCALE = 1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                                 milestones=[int(x*SCHED_TIMESCALE) for x in [6, 9, 12]], verbose=True)
energy_results = train_model(model=energy_model_0,
            train_dataloader=energy_train_loader, test_dataloader=energy_test_loader,
            optimizer=optimizer, loss_fn=loss_fn, device=device,
            seed=12, epochs=12)

# Danceability
optimizer = torch.optim.Adam(params=dance_model_0.parameters(), lr=0.01)
SCHED_TIMESCALE = 1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                                 milestones=[int(x*SCHED_TIMESCALE) for x in [6, 9, 12]], verbose=True)
dance_results = train_model(model=dance_model_0,
            train_dataloader=dance_train_loader, test_dataloader=dance_test_loader,
            optimizer=optimizer, loss_fn=loss_fn, device=device,
            seed=12, epochs=12)

# Valence
optimizer = torch.optim.Adam(params=valence_model_0.parameters(), lr=0.01)
SCHED_TIMESCALE = 1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1,
                                                 milestones=[int(x*SCHED_TIMESCALE) for x in [6, 9, 12]], verbose=True)
valence_results = train_model(model=valence_model_0,
            train_dataloader=valence_train_loader, test_dataloader=valence_test_loader,
            optimizer=optimizer, loss_fn=loss_fn, device=device,
            seed=12, epochs=15)


# Plot model loss curves
loss_curves(energy_results)
loss_curves(dance_results)
loss_curves(valence_results)

# Save models
torch.save(energy_model_0, f="saved items/energy_model_0.pt")
torch.save(dance_model_0, f="saved items/dance_model_0.pt")
torch.save(valence_model_0, f="saved items/valence_model_0.pt")