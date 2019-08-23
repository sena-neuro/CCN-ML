"""
Statistical useful notes:
(periodogram,AR,MUSIC)-(PSD)-(LR,MLPNN,SVM)


* Periodogram method
* Fourier Conversion
* FFT
* AR (auto-regressive) - (Burg or Levinson-Durbin algorithms)
* ARMA (auto-regressive moving-average)
* MUSIC (multiple signal classification) - PSD related
* STFT and wavelet (time-frequency compromise)
* zero padding process was performed
* Rectangular and Hanning windows
"""
import numpy as np
from sklearn.model_selection import train_test_split
# preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def preprocess(x, y, train_test=True, method="StandardScaler", scaler=True, transform=False):
    """
    Preprocessing steps: 
    1. MinMax Scaler
    2. Simple Transformer
    3. Train test split
"""
    if (method == "StandardScaler"):
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    # !!!! IMPORTANT FOR SVM
    elif (method == "MinMaxScaler"):
        scaler = MinMaxScaler(feature_range=(-1, 1))  # Standard scaler too
        scaler.fit(x)
        x = scaler.transform(x)

    # simple transformation which takes out the average
    else:
        data_mean = np.average(x)
        data_std = np.std(x)
        x = (x - data_mean) / data_std

    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        return x_train, x_test, y_train, y_test

    else:
        return x, y


# functions for Normalization
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


# ------------------------------------
# testing
if __name__ == "__main__":
    x = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    print("Trying preprocess()!\n")
    content = preprocess(x, y)
    if content != None:
        print("    preprocess() works.\n")
