
# ===========================
# gnss_wavelet_lstm_cnn_ensemble.py
# ===========================
# MIT License (c) 2025
# Purpose: Real-time GNSS spoofing detection feature pipeline and Keras models
# - SWT features (time-aligned) -> LSTM/GRU branch
# - CWT scalograms -> CNN branch
# - Ensemble fusion (late fusion with averaging, or learnable)
#
# Author: ChatGPT (adapted for your notebook)
# ===========================

import math
import numpy as np

# Optional deps: pywt for wavelets, tensorflow for Keras models
try:
    import pywt
except Exception as e:
    pywt = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, Input, Model
except Exception as e:
    tf = None
    layers = models = regularizers = Input = Model = None


# ---------------------------
# Utility: sliding windows
# ---------------------------
def sliding_windows_1d(x, win_len, hop_len):
    """
    Segment 1D array into overlapping windows [n_windows, win_len].
    Pads the tail with edge value if needed for last full window.
    """
    x = np.asarray(x).astype(np.float32).ravel()
    if win_len <= 0 or hop_len <= 0:
        raise ValueError("win_len and hop_len must be > 0")
    if len(x) < win_len:
        # pad to a single window
        pad = np.full(win_len - len(x), x[-1] if len(x) else 0.0, dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
    windows = []
    for start in range(0, len(x) - win_len + 1, hop_len):
        windows.append(x[start:start+win_len])
    if not windows:
        windows = [x[:win_len]]
    return np.stack(windows, axis=0)


# ---------------------------
# Wavelet features (SWT)
# ---------------------------
def swt_time_aligned_features(series, wavelet='db4', level=3, include_approx=True):
    """
    Stationary Wavelet Transform (SWT) features with preserved time alignment.
    Returns array of shape [T, F], where F = level (+1 if include_approx).
    Each column corresponds to detail coeff at a level (and optional approx at last level).
    """
    series = np.asarray(series).astype(np.float32).ravel()
    if pywt is None:
        raise ImportError("pywt (PyWavelets) is required for SWT features.")
    coeffs = pywt.swt(series, wavelet=wavelet, level=level, start_level=0, axis=-1)
    # coeffs: list of tuples (cA, cD) for each level; each same length as series
    # We will stack detail coefficients for all levels + final approximation if asked
    detail_list = []
    for (cA, cD) in coeffs:
        detail_list.append(cD.astype(np.float32))
    if include_approx:
        detail_list.append(coeffs[-1][0].astype(np.float32))
    features = np.stack(detail_list, axis=-1)  # [T, F]
    return features


# ---------------------------
# Wavelet scalogram (CWT)
# ---------------------------
def cwt_scalogram(series, scales=None, wavelet='morl', normalize=True):
    """
    Continuous Wavelet Transform scalogram.
    Returns array [n_scales, T], optionally normalized per-scale.
    """
    series = np.asarray(series).astype(np.float32).ravel()
    if pywt is None:
        raise ImportError("pywt (PyWavelets) is required for CWT scalograms.")
    if scales is None:
        # Default logarithmic scale range suitable for many sampling rates.
        scales = np.geomspace(2, 128, num=48).astype(np.float32)
    coef, freqs = pywt.cwt(series, scales, wavelet)
    scalogram = np.abs(coef).astype(np.float32)  # magnitude
    if normalize:
        # per-scale z-norm to reduce scale bias
        mu = scalogram.mean(axis=1, keepdims=True)
        sd = scalogram.std(axis=1, keepdims=True) + 1e-6
        scalogram = (scalogram - mu) / sd
    return scalogram  # [n_scales, T]


# ---------------------------
# Make streaming windows
# ---------------------------
def make_sequence_windows_from_swt(swt_features, win_len, hop_len):
    """
    swt_features: [T, F] -> windows [N, win_len, F]
    """
    T, F = swt_features.shape
    if T < win_len:
        # pad time dimension
        pad_rows = np.repeat(swt_features[[-1], :], win_len - T, axis=0)
        swt_features = np.vstack([swt_features, pad_rows])
        T = win_len
    starts = range(0, T - win_len + 1, hop_len)
    windows = [swt_features[s:s+win_len, :] for s in starts]
    if not windows:
        windows = [swt_features[:win_len, :]]
    return np.stack(windows, axis=0).astype(np.float32)


def make_image_windows_from_cwt(scalogram, win_len, hop_len):
    """
    scalogram: [H=n_scales, W=T] -> windows [N, H, win_len, 1]
    (Add channel dim for CNN.)
    """
    H, W = scalogram.shape
    if W < win_len:
        pad = np.repeat(scalogram[:, [-1]], win_len - W, axis=1)
        scalogram = np.concatenate([scalogram, pad], axis=1)
        W = win_len
    starts = range(0, W - win_len + 1, hop_len)
    imgs = [scalogram[:, s:s+win_len] for s in starts]
    if not imgs:
        imgs = [scalogram[:, :win_len]]
    imgs = np.stack(imgs, axis=0).astype(np.float32)  # [N, H, win_len]
    return imgs[..., np.newaxis]  # [N, H, win_len, 1]


# ---------------------------
# Keras models
# ---------------------------
def build_lstm_branch(input_timesteps, n_features, lstm_units=64, dropout=0.2):
    if tf is None:
        raise ImportError("TensorFlow is required to build the LSTM model.")
    x_in = Input(shape=(input_timesteps, n_features), name="swt_seq_input")
    x = layers.Masking(mask_value=0.0)(x_in)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    seq_out = layers.Dense(1, activation='sigmoid', name="seq_score")(x)
    return Model(inputs=x_in, outputs=seq_out, name="LSTM_branch")


def build_cnn_branch(img_height, img_width, channels=1, base_filters=32, dropout=0.2):
    if tf is None:
        raise ImportError("TensorFlow is required to build the CNN model.")
    i_in = Input(shape=(img_height, img_width, channels), name="cwt_img_input")
    x = layers.Conv2D(base_filters, (3,3), padding='same', activation='relu')(i_in)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(base_filters*2, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(base_filters*4, (3,3), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    img_out = layers.Dense(1, activation='sigmoid', name="img_score")(x)
    return Model(inputs=i_in, outputs=img_out, name="CNN_branch")


def build_ensemble_model(lstm_branch, cnn_branch, learnable_fusion=True):
    if tf is None:
        raise ImportError("TensorFlow is required to build the ensemble model.")
    seq_score = lstm_branch.output
    img_score = cnn_branch.output
    merged = layers.Concatenate(name="concat_scores")([seq_score, img_score])
    if learnable_fusion:
        x = layers.Dense(16, activation='relu')(merged)
        x = layers.LayerNormalization()(x)
        out = layers.Dense(1, activation='sigmoid', name="spoof_prob")(x)
    else:
        # simple average
        out = layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True), name="avg_spoof_prob")(merged)
    model = Model(inputs=[lstm_branch.input, cnn_branch.input], outputs=out, name="Ensemble_LSTM_CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC','Precision','Recall'])
    return model


# ---------------------------
# Example pipeline function
# ---------------------------
def make_training_tensors(series, win_len=128, hop_len=32,
                          swt_level=3, swt_wavelet='db4',
                          cwt_scales=None, cwt_wavelet='morl'):
    """
    Given a 1D time series, generate:
      - seq_windows: [N, win_len, F] for LSTM
      - img_windows: [N, H, win_len, 1] for CNN
    """
    # SWT features aligned with time
    swt_feats = swt_time_aligned_features(series, wavelet=swt_wavelet, level=swt_level, include_approx=True)  # [T, F]
    seq_windows = make_sequence_windows_from_swt(swt_feats, win_len, hop_len)  # [N, win_len, F]

    # CWT scalogram (time-frequency image)
    scalogram = cwt_scalogram(series, scales=cwt_scales, wavelet=cwt_wavelet, normalize=True)  # [H, T]
    img_windows = make_image_windows_from_cwt(scalogram, win_len, hop_len)  # [N, H, win_len, 1]

    return seq_windows, img_windows


# ---------------------------
# Lightweight TCN (optional)
# ---------------------------
def build_tcn_branch(input_timesteps, n_features, base_filters=32, stacks=3, dilations=(1,2,4,8), dropout=0.1):
    """
    A small causal dilated 1D CNN (TCN-like) as a faster alternative to LSTM.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build the TCN model.")
    x_in = Input(shape=(input_timesteps, n_features), name="swt_seq_input_tcn")
    x = x_in
    for s in range(stacks):
        for d in dilations:
            residual = x
            x = layers.Conv1D(base_filters, kernel_size=3, padding='causal', dilation_rate=d, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(base_filters, kernel_size=3, padding='causal', dilation_rate=d, activation='relu')(x)
            # match channels for residual if needed
            if residual.shape[-1] != x.shape[-1]:
                residual = layers.Conv1D(base_filters, kernel_size=1, padding='same')(residual)
            x = layers.Add()([x, residual])
            x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name="tcn_seq_score")(x)
    return Model(inputs=x_in, outputs=out, name="TCN_branch")


# ---------------------------
# Example usage (pseudo)
# ---------------------------
EXAMPLE = r"""
# Example (assuming you have numpy arrays 'signal' and binary labels 'y_win' per window):

from gnss_wavelet_lstm_cnn_ensemble import (
    make_training_tensors,
    build_lstm_branch, build_cnn_branch, build_ensemble_model
)

# 1) Feature generation
seq_windows, img_windows = make_training_tensors(signal, win_len=128, hop_len=32)

# 2) Models
lstm_branch = build_lstm_branch(input_timesteps=seq_windows.shape[1], n_features=seq_windows.shape[2])
cnn_branch  = build_cnn_branch(img_height=img_windows.shape[1], img_width=img_windows.shape[2])

# 3) Ensemble
ensemble = build_ensemble_model(lstm_branch, cnn_branch, learnable_fusion=True)

# 4) Train
ensemble.fit([seq_windows, img_windows], y_win, batch_size=64, epochs=20, validation_split=0.2)

# 5) Real-time: for each new chunk, compute same features and call ensemble.predict([...])
"""

if __name__ == "__main__":
    # Tiny self-check on synthetic data (no training to keep it lightweight)
    N = 2048
    t = np.linspace(0, 4*np.pi, N, dtype=np.float32)
    series = np.sin(3*t) + 0.2*np.random.randn(N).astype(np.float32)
    # Inject a transient (simulating spoof-like jump)
    series[900:910] += 3.0

    try:
        seq_windows, img_windows = make_training_tensors(series, win_len=128, hop_len=64)
        print("Seq windows:", seq_windows.shape, "Img windows:", img_windows.shape)
    except Exception as e:
        print("Feature generation skipped:", e)

    if tf is not None:
        try:
            lstm_branch = build_lstm_branch(seq_windows.shape[1], seq_windows.shape[2])
            cnn_branch  = build_cnn_branch(img_windows.shape[1], img_windows.shape[2])
            ensemble    = build_ensemble_model(lstm_branch, cnn_branch, learnable_fusion=True)
            print(ensemble.summary())
        except Exception as e:
            print("Model build skipped:", e)
    else:
        print("TensorFlow not available; models not built.")
