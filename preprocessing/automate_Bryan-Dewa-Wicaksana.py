"""
Automated Data Preprocessing Script
Nama Siswa: Bryan Dewa Wicaksana
Project: Membangun Sistem Machine Learning (MSML) Submission
Dataset: Heart Disease Prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def find_input_file(filename="heart_raw.csv"):
    """Mencari lokasi raw dataset secara dinamis."""
    possible_paths = [
        filename,
        os.path.join("dataset_raw", filename),
        os.path.join("..", "dataset_raw", filename),
        os.path.join("..", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"File raw dataset '{filename}' tidak ditemukan di lokasi: {possible_paths}")


def load_raw_data(file_path="heart_raw.csv"):
    """Memuat raw dataset dari file CSV."""
    real_path = find_input_file(file_path)
    df = pd.read_csv(real_path)
    print(f"[INFO] Raw dataset berhasil dimuat dari '{real_path}'. Shape: {df.shape}")
    return df


def clean_missing_and_duplicates(df):
    """Membersihkan nilai kosong dan duplikat dari dataset."""
    df_clean = df.copy()

    if df_clean.isnull().sum().sum() > 0:
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        print("[INFO] Missing values telah diimputasi.")

    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] {duplicate_count} baris duplikat berhasil dihapus.")

    return df_clean


def preprocess_data(df, target_col="target"):
    """
    Melakukan normalisasi/scaling pada fitur numerik 
    dan mengembalikan dataframe yang siap dilatih.
    """
    df_prep = clean_missing_and_duplicates(df)

    X = df_prep.drop(columns=[target_col])
    y = df_prep[target_col]

    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaling_cols = [c for c in continuous_cols if c in X.columns]

    scaler = StandardScaler()
    X[scaling_cols] = scaler.fit_transform(X[scaling_cols])

    df_processed = pd.concat([X, y.reset_index(drop=True)], axis=1)
    print(f"[INFO] Preprocessing selesai. Shape processed dataset: {df_processed.shape}")
    return df_processed, scaler


def run_pipeline(input_name="heart_raw.csv", output_name="heart_processed.csv"):
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    df_raw = load_raw_data(input_name)
    df_processed, _ = preprocess_data(df_raw, target_col="target")

    # Tentukan lokasi output
    if os.path.exists("preprocessing/dataset_preprocessing"):
        out_path = os.path.join("preprocessing", "dataset_preprocessing", output_name)
    elif os.path.exists("dataset_preprocessing"):
        out_path = os.path.join("dataset_preprocessing", output_name)
    else:
        out_path = output_name

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df_processed.to_csv(out_path, index=False)
    print(f"[SUCCESS] Processed dataset disimpan ke: {out_path}")
    return df_processed


if __name__ == "__main__":
    run_pipeline()
