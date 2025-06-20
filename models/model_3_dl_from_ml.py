# model_3_autoencoder_dl.py
"""
Model 3: ML‑driven Feature Extraction (Autoencoder) + Deep Learning Classification
EEG-based Schizophrenia vs Control
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# --- Configuration ---
DATA_DIR = 'data'                # Folder where raw EEG CSVs live
DEMOGRAPHIC_FILE = os.path.join(DATA_DIR, 'demographic.csv')
TRIAL_LENGTH = 6144              # samples per trial
FS = 1024                        # sampling rate (Hz)
BANDS = {
    "delta": (0.5, 4), "theta": (4, 8),
    "alpha": (8, 12), "beta": (12, 30),
    "gamma": (30, 100),
}

# --- Helper Functions ---
def load_demographic():
    df = pd.read_csv(DEMOGRAPHIC_FILE)
    df.columns = df.columns.str.strip()
    return df

def extract_manual_features(eeg_trial, fs=FS):
    """Manual features: stat, spectral, wavelet per channel"""
    feats = []
    # for each channel (shape: channels x time)
    for channel in eeg_trial:
        # Statistical
        feats += [
            np.mean(channel), np.std(channel),
            skew(channel), kurtosis(channel),
            np.min(channel), np.max(channel),
            np.median(channel)
        ]
        # Spectral via Welch
        freqs, psd = welch(channel, fs)
        for (low, high) in BANDS.values():
            idx = (freqs >= low) & (freqs <= high)
            feats.append(np.sum(psd[idx]))
        # Wavelet (db4, level=5)
        coeffs = pywt.wavedec(channel, 'db4', level=5)
        for c in coeffs:
            feats.append(np.mean(c))
            feats.append(np.std(c))
    return feats

def generate_feature_names(n_channels):
    names = []
    for ch in range(n_channels):
        names += [
            f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_skew', f'ch{ch}_kurt',
            f'ch{ch}_min', f'ch{ch}_max', f'ch{ch}_median'
        ]
        names += [f'ch{ch}_{band}_power' for band in BANDS]
        for lvl in range(6):
            names += [f'ch{ch}_w{lvl}_mean', f'ch{ch}_w{lvl}_std']
    return names

# --- Main Pipeline ---
def main():
    demo_df = load_demographic()
    all_feats, labels = [], []

    # Loop over subject folders
    for entry in os.listdir(DATA_DIR):
        if entry.endswith('.csv') and entry[:-4].isdigit():
            subj = int(entry[:-4])
            path = os.path.join(DATA_DIR, entry, f"{subj}.csv")
            if not os.path.exists(path):
                continue
            eeg_df = pd.read_csv(path, header=None)
            data = eeg_df.values[:, 4:]  # drop metadata cols
            n_trials = data.shape[0] // TRIAL_LENGTH
            data = data[:n_trials * TRIAL_LENGTH]
            trials = data.reshape(n_trials, TRIAL_LENGTH, -1)  # (n_trials, time, channels)
            for trial in trials:
                trial = trial.T  # (channels, time)
                feats = extract_manual_features(trial)
                all_feats.append(feats)
                lbl = demo_df.loc[demo_df['subject']==subj, 'group'].values[0]
                labels.append(lbl)

    # Assemble DataFrame
    n_channels = trials.shape[2]
    feature_names = generate_feature_names(n_channels)
    df_feat = pd.DataFrame(all_feats, columns=feature_names)
    df_feat['label'] = labels
    df_feat.to_csv('eeg_all_features.csv', index=False)
    print("✔️ Manual features extracted and saved to eeg_all_features.csv")

    # --- Autoencoder for Feature Compression ---
    X = df_feat.drop(columns=['label']).values
    y = df_feat['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build Autoencoder
    input_dim = X_train.shape[1]
    encoding_dim = 32

    inp = Input(shape=(input_dim,))
    enc = Dense(64, activation='relu')(inp)
    enc = Dense(encoding_dim, activation='relu')(enc)
    dec = Dense(64, activation='relu')(enc)
    dec = Dense(input_dim, activation='linear')(dec)

    autoenc = Model(inputs=inp, outputs=dec)
    autoenc.compile(optimizer='adam', loss='mse')
    autoenc.fit(
        X_train, X_train,
        epochs=50, batch_size=64, shuffle=True,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    print("✔️ Autoencoder trained")

    # Compress features
    encoder = Model(inputs=inp, outputs=enc)
    X_train_enc = encoder.predict(X_train)
    X_test_enc = encoder.predict(X_test)

    # --- Deep Learning Classifier on Encoded Features ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(encoding_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    history = model.fit(
        X_train_enc, y_train,
        validation_data=(X_test_enc, y_test),
        epochs=50, batch_size=64,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Evaluate
    loss, acc, auc = model.evaluate(X_test_enc, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
