import pandas as pd

def load_data(path):
    file = pd.read_csv(path)
    return file