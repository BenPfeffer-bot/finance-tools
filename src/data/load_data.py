import pandas as pd
import json

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    data.to_csv(file_path, index=False)

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f)
