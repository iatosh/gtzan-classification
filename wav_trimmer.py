import os

import librosa
import soundfile as sf


def trim_audio(file_path, duration=30):
    """
    指定された音声ファイルを30秒にトリミングし、元のファイルを上書きする関数
    """
    # ファイルが存在するか確認
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path} ")
        return

    try:
        # 音声ファイルを読み込む
        y, sr = librosa.load(file_path, sr=None)

        # トリミングするサンプル数を計算
        trim_samples = int(duration * sr)

        # 30秒にトリミング
        trimmed_audio = y[:trim_samples]

        # 元のファイルを上書き保存
        sf.write(file_path, trimmed_audio, sr)
        print(f"Trimmed: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
