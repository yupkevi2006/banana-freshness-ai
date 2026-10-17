import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_multimodal_predictor(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='linear', name='spoilage_predictor')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    img_feat_cols = [c for c in df.columns if c.startswith('img_feat_')]
    gas_cols = ['ppm_left_mean', 'ppm_right_mean', 'ppm_left_std', 'ppm_right_std', 'ppm_delta_mean']
    meta_cols = ['Day_Num', 'Temp1_C', 'Temp2_C']

    feature_cols = img_feat_cols + meta_cols + gas_cols
    target_cols = ['spoil_50', 'spoil_75', 'spoil_90', 'spoil_95']

    X = df[feature_cols].values
    y = df[target_cols].values

    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)

    return X, y, feature_cols, X_scaler

def run_training():
    csv_path = os.path.join("main", "extracted_data", "fused_multimodal_features.csv")
    X, y, feature_cols, X_scaler = prepare_data(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = build_multimodal_predictor(X.shape[1])

    # Custom training loop or proper target weighting so model learns day progression
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=4,
        verbose=0
    )

    bundle = {'model': model, 'scaler_X': X_scaler, 'feature_cols': feature_cols}
    with open(os.path.join("main", "model_bundle.pkl"), 'wb') as f:
        pickle.dump(bundle, f)

    model.save(os.path.join("main", "banana_ripeness_model.keras"))
    print("Bundle retrained and saved.")

if __name__ == "__main__":
    run_training()
