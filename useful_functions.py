import matplotlib.pyplot as plt
import random
import pandas.io.sql
import librosa

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def random_spectrogram(df: pandas.DataFrame):
    """Plots one randomly selected spectrogram from music_df."""
    x = random.randint(0,812)
    plot_spectrogram(df['Spectrogram'][x], f"{df['Song_Name'][x]} by {df['Artist_Name'][x]}\n"
                                                 f"ID: {df['Song_ID'][x]} | x: {x}")

def plot_loss_curves(train_loss, test_loss):
    """Plots train and test loss curves from the results dictionary"""
    loss = train_loss
    test_loss = test_loss

    epochs = range(len(train_loss))
    plt.figure(figsize=(15,7))

    #Plotting loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend();

    #Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.legend();

def random_song():
    rand_index = random.randrange(len(music_df_normal))
    song_name = music_df_normal.loc[rand_index]['Song_Name']
    artist_name = music_df_normal.loc[rand_index]['Artist_Name']
    return song_name, artist_name