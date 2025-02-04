import os
from pathlib import Path

import librosa
import soundfile as sf


def split_audio(input_path, output_dir, segment_length_sec):
    """
    音声ファイルを指定された秒数ごとに分割して保存する関数
    """
    try:
        # 音声ファイルを読み込む
        y, sr = librosa.load(input_path, sr=None)
        segment_length_samples = segment_length_sec * sr
        segment_count = 0

        # 分割処理
        for i, start in enumerate(range(0, len(y), segment_length_samples)):
            end = start + segment_length_samples
            if end > len(y):
                break  # 最後のセグメントが短い場合は無視
            segment = y[start:end]

            # 出力ファイル名とパスを生成
            base_name = Path(input_path).stem
            output_file = os.path.join(output_dir, f"{base_name}_{i + 1}.wav")
            sf.write(output_file, segment, sr)
            segment_count += 1

        print(f"Processed: {input_path} -> {output_dir}")
        return segment_count
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0


def process_genre_folder(input_genre_dir, output_genre_dir, segment_length_sec):
    """
    ジャンルフォルダ内のすべての音声ファイルを処理する関数
    """
    os.makedirs(output_genre_dir, exist_ok=True)
    total_files_processed = 0

    for file_name in os.listdir(input_genre_dir):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(input_genre_dir, file_name)

        # 音声ファイルを分割
        segment_count = split_audio(file_path, output_genre_dir, segment_length_sec)
        if segment_count > 0:
            total_files_processed += 1

    print(f"Completed processing genre folder: {input_genre_dir} -> {output_genre_dir}")
    return total_files_processed


def dataset_augmentation(input_dir, output_dir, segment_length_sec):
    """
    データセット全体を処理する関数
    """
    os.makedirs(output_dir, exist_ok=True)
    total_genres_processed = 0

    for genre in os.listdir(input_dir):
        genre_dir = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_dir):
            continue

        # 出力先のジャンルフォルダを作成
        output_genre_dir = os.path.join(output_dir, genre)
        files_processed = process_genre_folder(
            genre_dir, output_genre_dir, segment_length_sec
        )
        if files_processed > 0:
            total_genres_processed += 1

    print(f"Completed processing dataset: {input_dir} -> {output_dir}")
    return total_genres_processed
