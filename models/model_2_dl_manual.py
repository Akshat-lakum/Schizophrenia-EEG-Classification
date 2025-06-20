# model_2_dl_manual.py
"""
Model 2: Manual Feature Extraction + Deep Learning Classification
EEG-based Schizophrenia vs Control using a Keras feedforward network
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
DATA_DIR = 'data'
DEMOGRAPHIC_FILE = os.path.join(DATA_DIR, 'demographic.csv')
FEATURE_CSV = 'eeg_all_features.csv'
FS = 1024            # Sampling rate (Hz)
TRIAL_LENGTH = 6144  # Samples per trial
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 100)
}

# --- Utility Functions ---
def load_demographic():
    return pd.read_csv(DEMOGRAPHIC_FILE)

def generate_column_names(n_channels=70):
    names = []
    for ch in range(n_channels):
        names += [f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_skew', f'ch{ch}_kurt',
                  f'ch{ch}_min', f'ch{ch}_max', f'ch{ch}_median']
        names += [f'ch{ch}_{band}_power' for band in BANDS]
        for lvl in range(6):
            names += [f'ch{ch}_w{lvl}_mean', f'ch{ch}_w{lvl}_std']
    return names

def extract_features(eeg_trial, fs=FS):
    feats = []
    for channel in eeg_trial:
        # Statistical
        feats += [np.mean(channel), np.std(channel), skew(channel), kurtosis(channel),
                  np.min(channel), np.max(channel), np.median(channel)]
        # Spectral
        freqs, psd = welch(channel, fs)
        for band in BANDS.values():
            idx = (freqs >= band[0]) & (freqs <= band[1])
            feats.append(np.sum(psd[idx]))
        # Wavelet
        coeffs = pywt.wavedec(channel, 'db4', level=5)
        for c in coeffs:
            feats.append(np.mean(c))
            feats.append(np.std(c))
    return feats

# --- Main Pipeline ---
def main():
    # 1. Load data and extract features if CSV not exists
    demo = load_demographic()
    all_feats, labels, subj_ids = [], [], []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv') and fname[:-4].isdigit():
            subj = int(fname[:-4])
            path = os.path.join(DATA_DIR, fname, f"{subj}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, header=None)
            data = df.values[:, 4:]
            n_trials = data.shape[0] // TRIAL_LENGTH
            data = data[:n_trials * TRIAL_LENGTH]
            trials = data.reshape(n_trials, TRIAL_LENGTH, -1)
            for trial in trials:
                feat = extract_features(trial.T)
                all_feats.append(feat)
                labels.append(demo.loc[demo['subject']==subj, 'group'].values[0])
                subj_ids.append(subj)
    cols = generate_column_names(len(trials[0]))
    feat_df = pd.DataFrame(all_feats, columns=cols)
    feat_df['label'] = labels
    feat_df['subject'] = subj_ids
    feat_df.to_csv(FEATURE_CSV, index=False)
    print(f"Saved features to {FEATURE_CSV}")

    # 2. Prepare X and y
    X = feat_df.drop(['label', 'subject'], axis=1).values
    y = np.array(labels)
    # Encode labels if not numeric
    # y = LabelEncoder().fit_transform(y)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Build the DL model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 6. Callbacks
    es = EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True)
    ckpt = ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True)

    # 7. Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[es, ckpt],
        verbose=2
    )

    # 8. Evaluate
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {scores[0]:.4f}, Test Acc: {scores[1]:.4f}, Test AUC: {scores[2]:.4f}")

    # 9. Plots
    # Accuracy & Loss
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # ROC Curve
    y_proba = model.predict(X_test).ravel()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

    # Confusion Matrix
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Control','Schizophrenia'],
                yticklabels=['Control','Schizophrenia'])
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()
