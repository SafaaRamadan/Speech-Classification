import os
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from collections import defaultdict, Counter


def record(filename="sample.wav", fs=22050, duration=5):
    print("Recording in progress...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recording complete.")


def extractFeatures(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    y, _ = librosa.effects.trim(y, top_db=20)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfcc, axis=1)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=y))
    return np.hstack([mfcc_mean, spectral_centroid, spectral_rolloff, zero_crossing])


def load_data(path="C:/Users/raa_r/Downloads/free-spoken-digit-dataset/recordings"):
    features, labels = [], []
    for filename in os.listdir(path):
        if filename.startswith(('0_', '1_')):
            label = int(filename[0])
            file_path = os.path.join(path, filename)
            feature = extractFeatures(file_path)
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)


#Naive Bayes
def trainNB(X_train, y_train):
    classes = np.unique(y_train)
    model = {}
    for c in classes:
        X_c = X_train[y_train == c]
        model[c] = {
            'mean': X_c.mean(axis=0),
            'var': X_c.var(axis=0) + 1e-9,
            'prior': X_c.shape[0] / X_train.shape[0]
        }
    return model


def gaussianCTS(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1.0 / np.sqrt(2 * np.pi * var)) * exponent


def predictsingle(x, model):
    posteriors = {}
    for c, params in model.items():
        prior = np.log(params['prior'])
        likelihood = np.sum(np.log(gaussianCTS(x, params['mean'], params['var'])))
        posteriors[c] = prior + likelihood
    return max(posteriors, key=posteriors.get)


def predict(X, model):
    return np.array([predictsingle(x, model) for x in X])

def predictUI(model):
    print("\nRecord your voice (say 0 or 1)")
    record("user_input.wav")
    features = extractFeatures("user_input.wav").reshape(1, -1)
    pred = predict(features, model)
    print(f"\nThe model predicts: {pred[0]}")


# Bagging
def bagging(X_train, y_train, X_test, y_test, n_estimators=10):
    predictions = []
    for i in range(n_estimators):
        X_sample, y_sample = resample(X_train, y_train, replace=True, random_state=42)
        if i < 5:
            model = trainNB(X_sample, y_sample)
            y_pred = predict(X_test, model)
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X_test)
        predictions.append(y_pred)

    predictions = np.array(predictions)
    final_pred = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(predictions.shape[1])]

    print("Bagging Performance")
    evaluate(y_test, final_pred)



def evaluate(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_true, y_pred) * 100:.2f}%")
    print(f"F1 Score: {f1_score(y_true, y_pred) * 100:.2f}%")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- NB Performance ---")
    model_nb = trainNB(X_train, y_train)
    y_pred_nb = predict(X_test, model_nb)
    evaluate(y_test, y_pred_nb)

    print("\n--- Logistic Regression Performance---")
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    evaluate(y_test, y_pred_lr)

    print("\n--- Bagging Performance ---")
    bagging(X_train, y_train, X_test, y_test)

    choice = input("\nDo you want to predict your voice? (y/n): ").strip().lower()
    if choice == 'y':
        predictUI(model_nb)
