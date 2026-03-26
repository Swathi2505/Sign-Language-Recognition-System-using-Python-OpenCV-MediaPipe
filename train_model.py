import os
import pickle
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_DIR = "data"
MODEL_PATH = "sign_language_model.pkl"
NAMES_PATH = "sign_names.pkl"

def load_dataset(data_dir=DATA_DIR):
    X = []
    y = []
    sign_names = []
    # each subdirectory in data_dir is a sign name
    for i, sign in enumerate(sorted(os.listdir(data_dir))):
        sign_path = os.path.join(data_dir, sign)
        if not os.path.isdir(sign_path):
            continue
        sign_names.append(sign)
        for fname in sorted(os.listdir(sign_path)):
            if not fname.endswith(".pkl"):
                continue
            fpath = os.path.join(sign_path, fname)
            with open(fpath, "rb") as f:
                landmarks = pickle.load(f)
            X.append(landmarks)
            y.append(i)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, sign_names

def train_and_save():
    X, y, sign_names = load_dataset()
    if len(X) == 0:
        print("No data found in 'data' directory. Run collet_data.py to collect samples first.")
        return
    # split for a quick eval
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # build a pipeline with scaler + classifier (RandomForest supports predict_proba)
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")
    # save model and sign names
    joblib.dump(pipeline, MODEL_PATH)
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(sign_names, f)
    print(f"Saved model to {MODEL_PATH} and sign names to {NAMES_PATH}")
    print("Sign order (index -> label):")
    for idx, name in enumerate(sign_names):
        print(f"  {idx}: {name}")

if __name__ == "__main__":
    train_and_save()