# model_1_pipeline.py

import os
import pandas as pd
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ---------- Load Data ---------- #
print("Loading data...")
root_path = "../data/raw"
demo = pd.read_csv(f"{root_path}/demographic.csv")
demo.columns = demo.columns.str.strip()
time = pd.read_csv(f"{root_path}/time.csv")
columns = pd.read_csv(f"{root_path}/columnLabels.csv")

# ---------- Trial Extraction (example with subject 18) ---------- #
df_18 = pd.read_csv(f"{root_path}/18.csv/18.csv", header=None)
df_18.columns = columns.columns
df_18 = df_18.merge(time, on='sample', how='left')
subject_meta = demo[demo['subject'] == 18].iloc[0]
df_18['subject'] = 18
df_18['group'] = subject_meta['group']
df_18['age'] = subject_meta['age']
df_18['gender'] = subject_meta['gender']
df_18['education'] = subject_meta['education']

# ---------- Trial Matrix ---------- #
eeg_channels = columns.columns[4:]
trials = {}
for trial_num, trial_df in df_18.groupby('trial'):
    eeg_data = trial_df[eeg_channels].values.T
    trials[int(trial_num)] = eeg_data

print(f"Total trials: {len(trials)}")

# ---------- Feature Extraction ---------- #
def extract_features(trial_matrix):
    features = []
    for ch_data in trial_matrix:
        # Wavelet features
        coeffs = pywt.wavedec(ch_data, 'db4', level=4)
        wavelet_feats = [np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs]

        # Statistical features
        stat_feats = [np.mean(ch_data), np.std(ch_data), skew(ch_data), kurtosis(ch_data)]

        # Spectral features
        freqs, psd = welch(ch_data)
        spectral_feats = [np.mean(psd), np.std(psd)]

        features.extend(wavelet_feats + stat_feats + spectral_feats)
    return features

print("Extracting features...")
X, y = [], []
for trial_id, eeg in trials.items():
    feature_vec = extract_features(eeg)
    X.append(feature_vec)
    y.append(subject_meta['group'])

X = np.array(X)
y = np.array(y)

# ---------- Model Training & Evaluation ---------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

print("\nModel Comparison:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")
    print(classification_report(y_test, y_pred))

 
