import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout

patient_ids = ['100', '119', '208', '234']
window_size = 150
X_raw, y_binary = [], []

print("Fetching and processing multiple patients...")
for patient in patient_ids:
    record = wfdb.rdrecord(patient, pn_dir='mitdb')
    annotations = wfdb.rdann(patient, 'atr', pn_dir='mitdb')
    
    ecg_signal = record.p_signal[:, 0]
    for idx, label in zip(annotations.sample, annotations.symbol):
        if idx - window_size < 0 or idx + window_size >= len(ecg_signal): continue
            
        X_raw.append(ecg_signal[idx - window_size : idx + window_size])
        y_binary.append(0 if label == 'N' else 1)

X_raw = np.array(X_raw)[..., np.newaxis]
y_binary = np.array(y_binary)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

neg, pos = np.bincount(y_train)
class_weights = {0: (1/neg)*(len(y_train)/2.0), 1: (1/pos)*(len(y_train)/2.0)}

print("\nTraining 1D-CNN...")
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size * 2, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, class_weight=class_weights, verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Normal', 'Arrhythmia'], 
            yticklabels=['Normal', 'Arrhythmia'])
plt.title('Confusion Matrix: 1D-CNN (Optimized)', fontsize=12)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.tight_layout()
plt.show()
