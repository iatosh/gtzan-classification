import json
import os

import pandas as pd
from dataset_augmentator import dataset_augmentation
from feature_extractor import dataset_to_feature_csv
from model_trainer import train_multiple_models
from wav_trimmer import trim_audio
from wav_validator import check_and_report


# データに問題がないかチェック
def check_data():
    print("Checking data...")
    corrupted_files, duration_mismatch_files, output = check_and_report(
        "../data/genres"
    )
    print(output)
    return corrupted_files, duration_mismatch_files


# データの前処理
def preprocess(corrupted_files, duration_mismatch_files):
    # 破損している音声ファイルを削除
    if corrupted_files:
        print("\nRemoving corrupted files...")
        for file in corrupted_files:
            os.remove(file)

    # 30秒を超過している音声ファイルを30秒にトリミング
    if duration_mismatch_files:
        print("\nTrimming audio files...")
        for file, duration in duration_mismatch_files:
            if duration > 30:
                trim_audio(file)


# データセットの拡張
def augment_dataset():
    input_dir = "../data/genres"

    output_lengths = [3, 10]
    # 3秒と10秒のデータセットを作成
    for length in output_lengths:
        output_dir = f"../data/genres_{length}s"
        if not os.path.exists(output_dir):
            print(f"\nCreating {length}s dataset...")
            dataset_augmentation(input_dir, output_dir, length)


# 特徴量csvの作成
def make_feature_csvs():
    output_dir = "../data/genres_features"

    ## 3秒、10秒、30秒のデータセットの特徴量csvを作成
    feature_lengths = [3, 10, 30]
    input_dirs = {
        3: "../data/genres_3s",
        10: "../data/genres_10s",
        30: "../data/genres",
    }
    for length in feature_lengths:
        output_file = os.path.join(output_dir, f"features_{length}s.csv")
        if not os.path.exists(output_file):
            print(f"\nExtracting features from {length}s dataset...")
            input_dir = input_dirs[length]
            dataset_to_feature_csv(input_dir, output_dir, feature_length_sec=length)


# モデルの学習
def train_models(wandb_project_name, save_dir):
    # 各データセットの特徴量CSVファイルを読み込む
    df_3 = pd.read_csv("../data/genres_features/features_3s.csv")
    df_10 = pd.read_csv("../data/genres_features/features_10s.csv")
    df_30 = pd.read_csv("../data/genres_features/features_30s.csv")

    # ドロップアウト率のリスト
    dropout_rates = [0.2, 0.3, 0.4]
    # 隠れ層の次元数のリスト
    hidden_dims = [[256, 128, 64], [512, 256, 128, 64], [1024, 512, 256, 128, 64]]

    # 学習結果を格納する辞書
    results = {}
    # 3秒データセットでモデルを学習
    results.update(
        train_multiple_models(
            df_3,
            "3s",
            dropout_rates,
            hidden_dims,
            wandb_project_name,
            save_dir
        )
    )
    # 10秒データセットでモデルを学習
    results.update(
        train_multiple_models(
            df_10, "10s", dropout_rates, hidden_dims, wandb_project_name, save_dir
        )
    )
    # 30秒データセットでモデルを学習
    results.update(
        train_multiple_models(
            df_30, "30s", dropout_rates, hidden_dims, wandb_project_name, save_dir
        )
    )

    print("\nTraining complete.")

    return results


# モデルの評価
def model_evaluation(results, top_n=5):
    # Validation accuracyでソートして上位n個を表示
    sorted_val_results = sorted(
        results.items(), key=lambda x: x[1]["val_accuracy"], reverse=True
    )
    print(f"Top {top_n} models by validation accuracy:")
    for run_name, result in sorted_val_results[:top_n]:
        print(f"{run_name}: Validation accuracy: {result['val_accuracy']:.4f}")

    # Test accuracyでソートして上位n個を表示
    sorted_test_results = sorted(
        results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True
    )
    print(f"Top {top_n} models by test accuracy:")
    for run_name, result in sorted_test_results[:top_n]:
        print(f"{run_name}: Test accuracy: {result['test_accuracy']:.4f}")


def main():
    # 準備
    corrupted_files, duration_mismatch_files = check_data()
    preprocess(corrupted_files, duration_mismatch_files)
    augment_dataset()
    make_feature_csvs()

    # 学習
    save_dir = "../models"
    wandb_project_name = "nlp-final-report3"
    results = train_models(wandb_project_name, save_dir)
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 評価
    model_evaluation(results, top_n=5)


if __name__ == "__main__":
    main()
