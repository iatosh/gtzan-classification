import pandas as pd
import sklearn.model_selection as skms
import sklearn.preprocessing as skp


def generate_label_index(df):
    """
    ラベルとインデックスの対応を作成する関数
    """
    label_index = dict()
    for i, x in enumerate(df.label.unique()):
        label_index[x] = i
    return label_index


def label_to_index(df):
    """
    データフレームのラベルをインデックスに変換する関数
    """
    # ラベルとインデックスの対応を作成
    label_index = generate_label_index(df)
    # 元のデータフレームをコピー
    df_copy = df.copy()
    # ラベルをインデックスに変換
    df_copy.label = [label_index[label] for label in df_copy.label]
    return df_copy


def shuffle_and_split_data(df, seed=42):
    """
    データをシャッフルして学習用、検証用、評価用に分割する関数
    """
    # データをシャッフル
    df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 目的変数と説明変数に分割
    df_y = df_shuffle.pop("label")
    df_X = df_shuffle

    # データを学習用、検証用、評価用に分割
    X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(
        df_X, df_y, train_size=0.7, random_state=seed, stratify=df_y
    )
    X_dev, X_test, y_dev, y_test = skms.train_test_split(
        df_test_valid_X,
        df_test_valid_y,
        train_size=0.66,
        random_state=seed,
        stratify=df_test_valid_y,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def scale_X(X_train, X_dev, X_test):
    """
    特徴量をスケーリングする関数
    """
    scaler = skp.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    return X_train, X_dev, X_test


def preprocess_dataframe(df, seed=42):
    """
    データフレームを前処理する関数
    """
    # ラベルをインデックスに変換
    df_indexed = label_to_index(df)

    # データを分割
    X_train, X_dev, X_test, y_train, y_dev, y_test = shuffle_and_split_data(
        df_indexed, seed=seed
    )

    # 特徴量をスケーリング
    X_train, X_dev, X_test = scale_X(X_train, X_dev, X_test)

    return X_train, X_dev, X_test, y_train, y_dev, y_test
