import os
import wave


def check_wav_file(file_path):
    """
    指定されたWAVファイルが破損しているか、または30秒でないかどうかをチェックする関数
    """
    try:
        # WAVファイルを開く
        with wave.open(file_path, "r") as wav_file:
            # フレーム数とサンプルレートを取得
            num_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()

            # ファイルの長さ（秒）を計算
            duration = num_frames / float(frame_rate)

            # 長さが30秒でない場合
            if not (29.9 < duration < 30.1):  # 1秒未満の誤差を許容
                return "duration_mismatch", duration

        return None, None  # 問題なし

    except Exception:
        # エラーが発生したら破損とみなす
        return "corrupted", None


def check_wav_files_in_directory(directory_path):
    """
    指定されたディレクトリ内のすべてのWAVファイルをチェックし、破損しているものや30秒でないものを返す関数
    """
    corrupted_files = []
    duration_mismatch_files = []

    # ディレクトリ内の全ファイルを走査
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                issue, duration = check_wav_file(file_path) or (None, None)
                if issue == "corrupted":
                    corrupted_files.append(file_path)
                elif issue == "duration_mismatch":
                    duration_mismatch_files.append((file_path, duration))

    return corrupted_files, duration_mismatch_files


def check_and_report(directory_path):
    """
    指定されたディレクトリ内のWAVファイルをチェックし、結果を報告する関数
    """
    if not os.path.isdir(directory_path):
        return "Directory not found."

    corrupted_files, duration_mismatch_files = check_wav_files_in_directory(
        directory_path
    )

    output = ""
    if corrupted_files:
        output += "\nCorrupted:\n"
        for file in corrupted_files:
            output += f"{file}\n"
    else:
        output += "\nAll WAV files are intact.\n"

    if duration_mismatch_files:
        output += "\nDuration mismatch:\n"
        for file, duration in duration_mismatch_files:
            output += f"{file} (Length: {duration:.2f}s)\n"
    else:
        output += "\nAll WAV files have the correct duration.\n"

    return corrupted_files, duration_mismatch_files, output
