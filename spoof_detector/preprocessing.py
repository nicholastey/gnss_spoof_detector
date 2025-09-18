import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_no_diff(df, window_size=200, drop_extra=False):
    """
    Preprocess GNSS-SDR samples for LSTM input (no_diff or no_extra).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input GNSS samples with columns like prompt_i, prompt_q, cn0_db_hz, carrier_doppler_hz, etc.
    window_size : int
        Window size for rolling features.
    drop_extra : bool
        If True, drop less-informative columns (no_extra mode).
    
    Returns
    -------
    scaled_df : pd.DataFrame
        Preprocessed and standardized dataset ready for model input.
    """

    # Magnitude and phase
    df['prompt_magnitude'] = np.sqrt(df['prompt_i']**2 + df['prompt_q']**2)
    df['prompt_phase'] = np.arctan2(df['prompt_q'], df['prompt_i'])

    # Rolling statistics grouped by PRN
    df = df.sort_values(by=['prn', 'rx_time'])
    for col in ['cn0_db_hz', 'carrier_doppler_hz', 'prompt_magnitude']:
        df[f'{col}_roll_mean'] = df.groupby('prn')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df[f'{col}_roll_std'] = df.groupby('prn')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
        df[f'{col}_roll_min'] = df.groupby('prn')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).min()
        )
        df[f'{col}_roll_max'] = df.groupby('prn')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).max()
        )

    # Deltas for cn0 & doppler
    for col in ['cn0_db_hz', 'carrier_doppler_hz']:
        df[f'delta_{col}'] = df.groupby('prn')[col].diff().fillna(0)

    # Handle no_extra: drop redundant columns
    if drop_extra:
        df = df.drop(
            columns=['channel', 'fs', 'acq_doppler_step', 'pseudorange_m', 'acq_doopler_hz'],
            errors="ignore"
        )

    # Drop PRN (not for learning)
    df = df.drop(columns=['prn'], errors="ignore")

    # Scaling
    scaler = StandardScaler()
    features = df.drop(columns=['spoofed'], errors="ignore")
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns)

    if 'spoofed' in df.columns:
        scaled_df['spoofed'] = df['spoofed'].values

    return scaled_df
