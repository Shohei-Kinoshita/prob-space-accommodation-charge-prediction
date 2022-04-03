import sys
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from config import *
import preprocessing as pr
import train_predict as train_predict

warnings.filterwarnings('ignore')

# モデルリストの定義
DICT_MODEL_LIST = {
    'rf': RandomForestRegressor(random_state=0),
    'ridge': Ridge(random_state=0),
    'svr': SVR(),
    'neighbors': KNeighborsRegressor()
}


def main():
    args = sys.argv
    model_name = args[1]
    enc_type = args[2]
    model_save = args[3]

    print(f'model_name: {model_name}')
    print(f'enc_type: {enc_type}')
    print(f'monitoring: {model_save}')

    # データの読み込み
    print('read data...')
    df_train = pd.read_csv('input/train_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_test = pd.read_csv('input/test_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_train_station_info = pd.read_csv('input/train_data_distance_from_station.csv', dtype=DICT_DTYPES)
    df_test_station_info = pd.read_csv('input/test_data_distance_from_station.csv', dtype=DICT_DTYPES)
    df_train_name_features = pd.read_csv('input/train_data_name_features.csv')
    df_test_name_features = pd.read_csv('input/test_data_name_features.csv')

    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

    # 2020/4/30までの経過日数列を追加
    df_all = pr.create_elapsed_days(df_all)
    df_all.fillna(0, inplace=True)

    df_all = df_all[LIST_USE_COL]
    # エンコードタイプ(one-hot,label-enc)に合わせてエンコード
    df_all = pr.enc_categorical(df_all, LIST_ENC_COL, enc_type)

    # 各駅までの距離(km)を10次元に次元削減
    df_all_station_info = pd.concat([df_train_station_info, df_test_station_info], axis=0).reset_index(drop=True)
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=10, random_state=0))
    ])

    features_tmp = pipe.fit_transform(df_all_station_info)
    df_distance_features = pd.DataFrame(features_tmp, columns=[f'PCA_{i+1}' for i in range(features_tmp.shape[1])])

    df_train_distance_features = df_distance_features[:df_train.shape[0]].reset_index(drop=True)
    df_test_distance_features = df_distance_features[df_train.shape[0]:].reset_index(drop=True)

    # 使用する特徴量を全て結合
    X = df_all[:df_train.shape[0]].reset_index(drop=True)
    X = pd.concat([X, df_train_distance_features, df_train_name_features], axis=1)
    y = np.log1p(df_train[COL_Y])

    X_inference = df_all[df_train.shape[0]:].reset_index(drop=True)
    X_inference = pd.concat([X_inference, df_test_distance_features, df_test_name_features], axis=1)

    if enc_type == 'one-hot':
        scale = StandardScaler()
        scale.fit(X)
        X = pd.DataFrame(data=scale.transform(X), columns=X.columns)
        X_inference = pd.DataFrame(data=scale.transform(X_inference), columns=X_inference.columns)

    if model_save == 'True':
        # 学習、検証、モデルの保存
        df_oof = train_predict.fit_for_sklearn(X, y, DICT_MODEL_LIST, model_name, model_save=True)
        # 各foldで作成したモデルの予測平均を計算
        df_preds = train_predict.predict_fold_avg(X_inference, model_name)

        # out of foldの予測値を出力(スタッキングの特徴量として使用)
        df_oof.to_csv(f'input/train_{model_name}_out_of_fold.csv', index=False)
        df_preds.to_csv(f'input/test_{model_name}_out_of_fold.csv', index=False)
    else:
        # 学習、検証
        df_oof = train_predict.fit_for_sklearn(X, y, DICT_MODEL_LIST, model_name, model_save=False)

    print('finish')

if __name__ == '__main__':
    main()
