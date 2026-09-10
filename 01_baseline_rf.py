import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Fetching data for patient 100...")
record = wfdb.rdrecord('100', pn_dir='mitdb')
annotations = wfdb.rdann('100', 'atr', pn_dir='mitdb')

ecg_signal = record.p_signal[:, 0]
beat_indices = annotations.sample
labels = annotations.symbol

window_size = 150
features_list = []
binary_labels = []

print("Extracting FFT features...")
for i in range(len(beat_indices)):
    idx = beat_indices[i]
    label = labels[i]
    
    if idx - window_size < 0 or idx + window_size >= len(ecg_signal): continue
        
    beat_window = ecg_signal[idx - window_size : idx + window_size]
    binary_label = 0 if label == 'N' else 1
    
    # FFT
    N_samples = len(beat_window)
    fft_magnitudes = np.abs(fft(beat_window))[0:N_samples//2]
    frequencies = fftfreq(N_samples, 1 / record.fs)[0:N_samples//2]
    
    total_energy = np.sum(fft_magnitudes ** 2)
    low_freq_energy = np.sum(fft_magnitudes[frequencies <= 10] ** 2)
    
    features_list.append({
        'dominant_freq': frequencies[np.argmax(fft_magnitudes)],
        'total_energy': total_energy,
        'energy_ratio': low_freq_energy / (total_energy - low_freq_energy + 1e-5)
    })
    binary_labels.append(binary_label)

df_features = pd.DataFrame(features_list)
df_features['label'] = binary_labels

X_train, X_test, y_train, y_test = train_test_split(
    df_features.drop('label', axis=1), df_features['label'], 
    test_size=0.2, random_state=42, stratify=df_features['label']
)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Arrhythmia']))

# Extract TP, TN, FP, FN
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n--- Confusion Matrix Breakdown ---")
print(f"True Negatives (Correctly identified normal beats): {tn}")
print(f"False Positives (False alarms): {fp}")
print(f"False Negatives (Missed arrhythmias): {fn}")
print(f"True Positives (Correctly identified arrhythmias): {tp}")

# Visual Table (Heatmap)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Arrhythmia'], 
            yticklabels=['Normal', 'Arrhythmia'])
plt.title('Confusion Matrix: Random Forest (Baseline)', fontsize=12)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.tight_layout()
plt.show()
