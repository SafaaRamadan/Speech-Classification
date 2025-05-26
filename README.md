# Speech-Digit-Classifier-
A machine learning-based voice classifier that distinguishes between the spoken digits 0 and 1 using:
  - Gaussian Naive Bayes (from scratch)
  -  Logistic Regression
  -  Bagging (Naive Bayes + Logistic Regression)
----
## Features
- Record your voice in real-time
- Preprocess audio: silence removal, normalization, pre-emphasis
- Extract features: 
  - MFCCs (Mel-frequency cepstral coefficients)
  - Spectral Centroid
  - Spectral Rolloff
  - Zero Crossing Rate
- Train and evaluate:
  - Gaussian Naive Bayes (custom implementation)
  - Logistic Regression (scikit-learn)
  - Bagging 
- Metrics: Accuracy, Precision, Recall, F1-Score
---
## Dataset
This project uses the Free Spoken Digit Dataset (FSDD), an open dataset with 
                      
                      .wav 
  recordings of spoken digits by multiple speakers.
To download the dataset:
   Open your terminal and run:
   
       git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
