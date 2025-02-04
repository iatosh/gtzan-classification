import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_statistics(feature):
    """
    特徴量の統計量を計算する関数
    """
    mean = np.mean(feature)  # 平均
    var = np.var(feature)  # 分散
    min_val = np.min(feature)  # 最小値
    max_val = np.max(feature)  # 最大値
    return pd.Series(
        [mean, var, min_val, max_val],
        index=["mean", "var", "min", "max"],
    )


def extract_features(file_path):
    """
    音声ファイルから全ての特徴量を抽出し、統計量を計算する関数
    """
    # 音声ファイルを読み込む
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Error loading file: {file_path}, error: {e}")
        return None

    # 特徴量を格納する辞書
    features = {}

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for i, row in enumerate(chroma):  # 各音階ごとに統計量を計算
        stats = calculate_statistics(row)
        for stat_name, value in stats.items():
            features[f"chroma_{pitch_names[i]}_{stat_name}"] = value

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for i, row in enumerate(mfcc):  # 各MFCC係数ごとに統計量を計算
        stats = calculate_statistics(row)
        for stat_name, value in stats.items():
            features[f"mfcc{i}_{stat_name}"] = value

    # RMS
    rms = librosa.feature.rms(y=y).flatten()
    stats = calculate_statistics(rms)
    for stat_name, value in stats.items():
        features[f"rms_{stat_name}"] = value

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    stats = calculate_statistics(spectral_bandwidth)
    for stat_name, value in stats.items():
        features[f"spectral_bandwidth_{stat_name}"] = value

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    stats = calculate_statistics(spectral_centroid)
    for stat_name, value in stats.items():
        features[f"spectral_centroid_{stat_name}"] = value

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, row in enumerate(spectral_contrast):  # 各周波数帯域ごとに統計量を計算
        stats = calculate_statistics(row)
        for stat_name, value in stats.items():
            features[f"spectral_contrast{i}_{stat_name}"] = value

    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y).flatten()
    stats = calculate_statistics(spectral_flatness)
    for stat_name, value in stats.items():
        features[f"spectral_flatness_{stat_name}"] = value

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
    stats = calculate_statistics(spectral_rolloff)
    for stat_name, value in stats.items():
        features[f"spectral_rolloff_{stat_name}"] = value

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_names = ["5x", "5y", "m3x", "m3y", "M3x", "M3y"]
    for i, row in enumerate(tonnetz):  # 各Tonnetz特徴ごとに統計量を計算
        stats = calculate_statistics(row)
        for stat_name, value in stats.items():
            features[f"tonnetz_{tonnetz_names[i]}_{stat_name}"] = value

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).flatten()
    stats = calculate_statistics(zero_crossing_rate)
    for stat_name, value in stats.items():
        features[f"zero_crossing_rate_{stat_name}"] = value

    # Tempo
    tempo = librosa.feature.tempo(y=y, sr=sr)
    features["tempo"] = tempo[0]  # tempoは数値なのでそのまま

    return pd.Series(features)


def process_file(file_path):
    """
    指定された音声ファイルから特徴量を抽出し、ラベルを付与する関数
    """
    try:
        # 音声ファイルから特徴量を抽出
        features = extract_features(file_path)
        # 特徴量の抽出に失敗した場合、Noneを返す
        if features is None:
            return None
        # ファイルパスからラベル（ディレクトリ名）を取得
        label = os.path.basename(os.path.dirname(file_path))
        # 特徴量にラベルを追加
        features["label"] = label
        # 特徴量とラベルを含むSeriesを返す
        return features
    except Exception as e:
        # エラーが発生した場合、エラーメッセージを出力し、Noneを返す
        print(f"Error processing file: {file_path}, error: {e}")
        return None

def dataset_to_feature_csv(input_dir, output_dir, feature_length_sec=None):
    """
    ディレクトリ内の全ての音声ファイルから特徴量を抽出し、CSVファイルに保存する関数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    all_features = []  # 特徴量を格納するリスト
    error_files = []  # エラーファイル記録用
    for file_path in tqdm(file_paths):
        result = process_file(file_path)
        if result is not None:
            all_features.append(result)
        else:
            error_files.append(file_path)  # エラーが発生したファイルを記録

    # DataFrameに変換してCSVに保存
    df = pd.DataFrame(all_features)
    output_path = (
        os.path.join(output_dir, f"features_{feature_length_sec}s.csv")
        if feature_length_sec
        else os.path.join(output_dir, "features.csv")
    )
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}, shape: {df.shape}")

    if error_files:  # エラーファイルが存在する場合
        error_file_path = os.path.join(output_dir, "error_files.txt")
        with open(error_file_path, "w") as f:
            for file in error_files:
                f.write(f"{file}\n")
        print(f"Error files are recorded in {error_file_path}")
