import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from data_preperation import preprocess_dataframe
from early_stopping_pytorch import EarlyStopping
from model_evaluator import evaluate_model
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# PyTorchのバージョンを表示
print("Torch version:-", torch.__version__)

# 乱数シードを設定
SEED = 42
torch.manual_seed(SEED)

INPUT_DIMENSION = 205  # 特徴量ベクトルの次元数


def train_model(
    model,
    epochs,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_dev,
    y_dev,
    save_path,
    device="cpu",
):
    # Metalが利用可能かどうかを確認し、利用可能であればMetalを、そうでなければCPUを使用
    print(f"Using device: {device}")
    model.to(device)

    # EarlyStoppingを初期化
    early_stopping = EarlyStopping(patience=7, verbose=True, path=save_path)

    # バッチサイズを設定
    batch_size = 128

    # wandbにバッチサイズを記録
    wandb.config.batch_size = batch_size

    # 学習データと検証データをTensorDatasetに変換
    train_dataset = TensorDataset(
        torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32),
        torch.tensor(y_train.values.astype(np.int64), dtype=torch.long),
    )
    # 学習データローダーを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TensorDataset(
        torch.tensor(X_dev.values.astype(np.float32), dtype=torch.float32),
        torch.tensor(y_dev.values.astype(np.int64), dtype=torch.long),
    )
    # 検証データローダーを作成
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # エポック数だけ学習を繰り返す
    for epoch in range(epochs):
        # モデルを学習モードに設定
        model.train()
        # 学習損失を初期化
        train_loss = 0.0
        # 学習データの正解数を初期化
        correct_train = 0
        # 学習データの総数を初期化
        total_train = 0
        # 学習データローダーからバッチを取り出して学習
        for batch_X, batch_y in train_loader:
            # データをGPUに転送
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # 勾配を初期化
            optimizer.zero_grad()
            # モデルに入力して出力を得る
            outputs = model(batch_X)
            # 損失を計算
            loss = criterion(outputs, batch_y)
            # 誤差逆伝播
            loss.backward()
            # パラメータ更新
            optimizer.step()
            # 学習損失を累積
            train_loss += loss.item()
            # 予測結果を取得
            predicted = torch.argmax(outputs.data, 1)
            # 学習データの総数を更新
            total_train += batch_y.size(0)
            # 学習データの正解数を更新
            correct_train += (predicted == batch_y).sum().item()

        # 学習精度を計算
        train_accuracy = correct_train / total_train
        # 学習損失を平均化
        train_loss /= len(train_loader)

        # モデルを評価モードに設定
        model.eval()
        # 検証損失を初期化
        val_loss = 0.0
        # 検証データの正解数を初期化
        correct_val = 0
        # 検証データの総数を初期化
        total_val = 0
        # 勾配計算をしない
        with torch.no_grad():
            # 検証データローダーからバッチを取り出して評価
            for batch_X, batch_y in dev_loader:
                # データをGPUに転送
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                # モデルに入力して出力を得る
                outputs = model(batch_X)
                # 損失を計算
                loss = criterion(outputs, batch_y)
                # 検証損失を累積
                val_loss += loss.item()
                # 予測結果を取得
                predicted = torch.argmax(outputs.data, 1)
                # 検証データの総数を更新
                total_val += batch_y.size(0)
                # 検証データの正解数を更新
                correct_val += (predicted == batch_y).sum().item()

        # 検証精度を計算
        val_accuracy = correct_val / total_val
        # 検証損失を平均化
        val_loss /= len(dev_loader)

        # ログを記録
        wandb.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch,
            }
        )

        # EarlyStoppingを適用
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def create_model(hidden_dims, dropout_rate=0.2):
    """
    指定された隠れ層の次元とドロップアウト率に基づいて、ニューラルネットワークモデルを作成する関数
    """
    layers = []
    input_dim = INPUT_DIMENSION  # 入力層の次元数を設定
    for hidden_dim in hidden_dims:
        # 線形層を追加
        layers.append(nn.Linear(input_dim, hidden_dim))
        # ReLU活性化関数を追加
        layers.append(nn.ReLU())
        # ドロップアウト層を追加 (dropout_rateが0より大きい場合)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        # 次の層への入力次元を更新
        input_dim = hidden_dim
    # 出力層を追加 (10クラス分類)
    layers.append(nn.Linear(input_dim, 10))
    # レイヤーをまとめたSequentialモデルを返す
    return nn.Sequential(*layers)


def train_and_evaluate_model(model, df, save_path, epochs=200, device="cpu"):
    # データフレームを前処理
    X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_dataframe(df, seed=SEED)

    # 最適化アルゴリズムを定義
    optimizer = optim.Adam(model.parameters())
    # 損失関数を定義
    criterion = nn.CrossEntropyLoss()

    # モデルの概要を表示
    summary(model)

    # モデルを学習
    train_model(
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        device=device,
        save_path=save_path,
    )

    return model, X_test, y_test


def load_model(model_dir, hidden_dim, dropout_rate, save_dir, device="cpu"):
    """
    保存されたモデルをロードする関数
    """
    model = create_model(hidden_dim, dropout_rate=dropout_rate)
    model_path = os.path.join(save_dir, model_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    model.to(device)
    return model


def train_single_model(
    df, dataset_identifier, dropout_rate, hidden_dim, wandb_project_name, save_dir
):
    """
    単一のモデルを学習させる関数
    """
    print(f"Training model with dropout rate: {dropout_rate}, hidden dim: {hidden_dim}")

    # Metalが利用可能かどうかを確認し、利用可能であればMetalを、そうでなければCPUを使用
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルを作成
    model = create_model(hidden_dim, dropout_rate=dropout_rate)

    # wandbのrun名を生成
    run_name = f"{dataset_identifier}_dropout_{dropout_rate}_hidden_{hidden_dim[0]}"

    # wandbの初期化
    config = {
        "dropout_rate": dropout_rate,
        "hidden_dim": hidden_dim,
        "dataset_identifier": dataset_identifier,
    }
    wandb.init(project=wandb_project_name, name=run_name, config=config)
    wandb.watch(model)

    # モデルと評価データを保存するディレクトリを作成
    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)

    # モデルを学習・評価
    model, X_test, y_test = train_and_evaluate_model(
        model, df, f"{save_dir}/{run_name}/model.pth", device=device
    )

    # 保存先のディレクトリを作成
    os.makedirs(f"{save_dir}/{run_name}", exist_ok=True)

    # 評価用データを保存
    np.savez(
        f"{save_dir}/{run_name}/test_data.npz",
        X_test=X_test,
        y_test=y_test,
    )

    # モデルを評価
    best_param_model = load_model(
        run_name, hidden_dim, dropout_rate, save_dir, device=device
    )  # val_accuracyが最も高かった時のパラメータをロードしたモデル
    test_dataset = TensorDataset(
        torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32),
        torch.tensor(y_test.values.astype(np.int64), dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(
        best_param_model, test_loader, device, criterion
    )

    # wandbから最終的な検証精度を取得
    val_accuracy = wandb.run.summary.get("val_accuracy")
    # wandbにテスト精度と損失を記録
    wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss})

    wandb.finish()
    return {run_name: {"val_accuracy": val_accuracy, "test_accuracy": test_accuracy}}


def train_multiple_models(
    df, dataset_identifier, dropout_rates, hidden_dims, wandb_project_name, save_dir
):
    """
    複数のモデルを異なるドロップアウト率と隠れ層の次元で学習させる関数
    """
    results = {}
    for dropout_rate in dropout_rates:
        for hidden_dim in hidden_dims:
            results.update(
                train_single_model(
                    df,
                    dataset_identifier,
                    dropout_rate,
                    hidden_dim,
                    wandb_project_name,
                    save_dir,
                )
            )
    return results
