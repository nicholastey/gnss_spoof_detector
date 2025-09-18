import socket
import numpy as np
import pandas as pd
from tensorflow import keras
from collections import defaultdict, deque
from preprocessing import preprocess_no_diff   # the function I gave you earlier

HOST = '127.0.0.1'
PORT = 5736
TIMESTEPS = 50          # must match what you trained LSTM on
WINDOW_SIZE = 200       # rolling window for feature engineering
FEATURE_COLUMNS = [
    'channel','prn','acq_doopler_hz','acq_doppler_step','fs',
    'prompt_i','prompt_q','cn0_db_hz','carrier_doppler_hz',
    'pseudorange_m','rx_time'
]

# Load LSTM models
lstm_no_diff = keras.models.load_model('lstm_model_no_diff_new.h5', compile=False)
lstm_no_extra = keras.models.load_model('lstm_model_no_extra.h5', compile=False)

# Rolling buffer: keep last WINDOW_SIZE samples per PRN
buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# Start socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()
client_socket, addr = server_socket.accept()
print("Connected by", addr)

timestep = 0
while True:
    data = client_socket.recv(569)
    if not data:
        break

    try:
        values = list(map(float, data.decode().split(', ')[:-1]))
    except:
        continue

    # Parse into per-channel dict (8 channels, 11 features each)
    records = []
    for i in range(8):
        row = values[11*i:11*(i+1)]
        record = dict(zip(FEATURE_COLUMNS, row))
        records.append(record)

    df_new = pd.DataFrame(records)

    # Update buffers per PRN
    for _, row in df_new.iterrows():
        prn = int(row['prn'])
        buffers[prn].append(row)

        # Only process once we have enough samples
        if len(buffers[prn]) >= TIMESTEPS:
            buffer_df = pd.DataFrame(buffers[prn])

            # ----- Preprocessing -----
            processed_no_diff = preprocess_no_diff(buffer_df.copy(), window_size=WINDOW_SIZE, drop_extra=False)
            processed_no_extra = preprocess_no_diff(buffer_df.copy(), window_size=WINDOW_SIZE, drop_extra=True)

            # Align shapes for LSTM
            X_no_diff = processed_no_diff.drop(columns=['spoofed'], errors='ignore').values
            X_no_extra = processed_no_extra.drop(columns=['spoofed'], errors='ignore').values

            # Use only last TIMESTEPS rows for prediction
            X_no_diff = X_no_diff[-TIMESTEPS:].reshape((1, TIMESTEPS, X_no_diff.shape[1]))
            X_no_extra = X_no_extra[-TIMESTEPS:].reshape((1, TIMESTEPS, X_no_extra.shape[1]))

            # ----- LSTM Predictions -----
            pred_no_diff = lstm_no_diff.predict(X_no_diff, verbose=0)
            pred_no_extra = lstm_no_extra.predict(X_no_extra, verbose=0)

            # Decision logic (example: binary classification, spoofed prob > 0.5)
            if pred_no_diff[0,1] > 0.5 or pred_no_extra[0,1] > 0.5:
                print(f"[Time:{timestep}] Suspicious Signal on PRN {prn} | "
                      f"no_diff={pred_no_diff[0,1]:.3f}, no_extra={pred_no_extra[0,1]:.3f}")

    timestep += 1

client_socket.close()
server_socket.close()
