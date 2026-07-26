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
from sklearn.model_selection import train_test_split


def load_raw_data(file_path="heart_raw.csv"):
    """Memuat raw dataset dari file CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File raw dataset '{file_path}' tidak ditemukan.")
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw dataset berhasil dimuat. Shape: {df.shape}")
    return df


def clean_missing_and_duplicates(df):
    """Membersihkan nilai kosong dan duplikat dari dataset."""
    df_clean = df.copy()

    # Tangani missing value jika ada
    if df_clean.isnull().sum().sum() > 0:
        # Isi nilai numerik kosong dengan median per kolom
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        print("[INFO] Missing values telah diimputasi.")

    # Tangani duplikat jika ada
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

    # Pisahkan fitur dan target
    X = df_prep.drop(columns=[target_col])
    y = df_prep[target_col]

    # Kolom numerik kontinu yang perlu di-scale
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaling_cols = [c for c in continuous_cols if c in X.columns]

    scaler = StandardScaler()
    X[scaling_cols] = scaler.fit_transform(X[scaling_cols])

    # Gabungkan kembali fitur terproses dengan target
    df_processed = pd.concat([X, y.reset_index(drop=True)], axis=1)
    print(f"[INFO] Preprocessing selesai. Shape processed dataset: {df_processed.shape}")
    return df_processed, scaler


def run_pipeline(input_path="heart_raw.csv", output_path="heart_processed.csv"):
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    df_raw = load_raw_data(input_path)
    df_processed, _ = preprocess_data(df_raw, target_col="target")
    df_processed.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed dataset disimpan ke: {output_path}")
    return df_processed


if __name__ == "__main__":
    run_pipeline()
