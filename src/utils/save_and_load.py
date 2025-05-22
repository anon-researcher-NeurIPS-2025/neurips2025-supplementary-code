import pickle
import os

def save_data(data, filename="trajectories.pkl"):
    """ Saves the data to a .pkl file, creating directories if needed """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {filename}")

def load_data(filename="trajectories.pkl"):
    """ Loads the data from a .pkl file """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded from {filename}")
    return data