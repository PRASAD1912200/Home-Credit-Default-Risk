import pandas as pd

def load_data():
    print("Loading datasets...")
    train = pd.read_csv('Data/application_train.csv')
    test = pd.read_csv('Data/application_test.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test
