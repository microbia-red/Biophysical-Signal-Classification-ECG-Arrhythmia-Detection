# Biophysical Signal Classification: ECG Arrhythmia Detection

## Overview
This repository explores the classification of arrhythmias in 1D Electrocardiogram (ECG) signals using the PhysioNet MIT-BIH Database. It highlights the critical difference between algorithm selection and data curation in HealthTech by comparing a classical Machine Learning approach (FFT feature engineering) with a data-centric Deep Learning architecture (1D-CNN).

## Repository Structure
- `01_baseline_rf.py`: Establishes a baseline using a Random Forest classifier fed with frequency-domain features (Fast Fourier Transform). 
- `02_optimized_cnn.py`: The optimized approach using a 1D Convolutional Neural Network trained on raw time-series data across a curated multi-patient dataset.

## Installation
```bash
pip install wfdb numpy pandas scipy matplotlib seaborn scikit-learn tensorflow
```

## Methodology & Algorithmic Comparison

### 1. The Baseline: Classical ML & FFT (`01_baseline_rf.py`)
- **Approach:** Transitioned raw ECG windows into the frequency domain using FFT. Extracted biophysical markers like dominant frequency and energy band ratios.
- **Result:** The Random Forest achieved a high overall accuracy but completely failed on the minority class (**0% Recall** for arrhythmias) due to severe class imbalance in a single-patient sample.

### 2. The Solution: Data-Centric 1D-CNN (`02_optimized_cnn.py`)
- **Approach:** Shifted to Deep Learning to allow the model to learn morphological patterns directly from the raw time-series. More importantly, adopted a **data-centric strategy** by aggregating data from multiple high-risk patients (119, 208, 234) and applying dynamic class weighting.
- **Result:** The 1D-CNN successfully identified the underlying pathology, achieving an **87.8% Recall** for critical arrhythmias while maintaining a low False Positive rate.

## Conclusion
This project demonstrates that in medical time-series analysis, sophisticated feature engineering (FFT) and algorithmic choice cannot overcome a fundamental lack of pathological data. Curing the dataset for clinical representation was the definitive factor in building a viable detection model.

## Future Work
- **Threshold Tuning:** Lowering the classification decision boundary to prioritize Sensitivity over Specificity.
- **Time-Series Data Augmentation:** Applying time-warping or stochastic noise injection to artificially increase the minority class.
- **Advanced Architecture:** Appending an LSTM layer to the CNN to provide sequential context between consecutive beats.
