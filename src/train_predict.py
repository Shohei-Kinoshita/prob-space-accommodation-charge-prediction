import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from numpy.ma.core import argsort
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def fit_for_sklearn(X, y, dict_model_list, model_name, model_save=True):
    """
    KFoldによる学習、検証
    out of foldによる予測をデータフレームで返す
    """
    scores = []
    preds = []
    va_idxes = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        print('='*30)
        print(f'fold: {i}')
        X_train, X_valid = X.loc[train_idx, :], X.loc[valid_idx, :]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = dict_model_list[model_name]
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        score = np.sqrt(mean_squared_error(y_valid, pred))
        print(f'score: {score}')
        scores.append(score)
        preds.append(pred)
        va_idxes.append(valid_idx)

        if model_save:
            filename = f'models/{model_name}_fold{i}.pkl'
            pickle.dump(model, open(filename, 'wb'))

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = argsort(va_idxes)
    df_oof = pd.DataFrame(preds[order], columns=[f'{model_name}_stacking'])
    rmsle_mean = np.mean(scores)
    print('='*30)
    print(f'rmsle_mean: {rmsle_mean}')
    print(f'{model_name}_oof')
    print(df_oof.head())
    return df_oof


def fit_for_lgbm(X, y, model_name, params, verbose=True, model_save=True):
    """
    KFoldによる学習、検証
    out of foldによる予測をデータフレームで返す
    """
    scores = []
    preds = []
    va_idxes = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        print('='*50)
        print(f'fold: {i}')
        X_train, X_valid = X.loc[train_idx, :], X.loc[valid_idx, :]
        y_train, y_valid = y[train_idx], y[valid_idx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        lgb_results = {}
        model = lgb.train(params=params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['Train', 'Test'],
                        num_boost_round=300,
                        early_stopping_rounds=20,
                        evals_result=lgb_results,
                        feval=calc_rsmle,
                        verbose_eval=verbose)

        score = np.sqrt(mean_squared_error(y_valid, model.predict(X_valid)))
        print(f'mean_squared_log_error: {score}')
        pred = model.predict(X_valid)
        scores.append(score)
        preds.append(pred)
        va_idxes.append(valid_idx)
        
        if model_save:
            filename = f'models/{model_name}_fold{i}.pkl'
            pickle.dump(model, open(filename, 'wb'))

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = argsort(va_idxes)
    df_oof = pd.DataFrame(preds[order], columns=[f'{model_name}_stacking'])
    rmsle_mean = np.mean(scores)
    print('='*50)
    print(f'rmsle_mean: {rmsle_mean}')
    print(f'{model_name}_oof')
    print(df_oof.head())
    return df_oof

def predict_fold_avg(X_inference, model_name):
    """
    テストデータに対する予測
    5foldで作成したモデルのそれぞれの予測と平均をデータフレームで返す
    """
    list_preds_tmp = []
    for i in range(1, 6):
        model_path = f'models/{model_name}_fold{i}.pkl'
        model = pickle.load(open(model_path, 'rb'))
        pred = model.predict(X_inference)
        list_preds_tmp.append(pred)
    df_preds = pd.DataFrame({'model_1': np.squeeze(list_preds_tmp[0]),
                             'model_2': np.squeeze(list_preds_tmp[1]),
                             'model_3': np.squeeze(list_preds_tmp[2]),
                             'model_4': np.squeeze(list_preds_tmp[3]),
                             'model_5': np.squeeze(list_preds_tmp[4])})
    df_preds[f'{model_name}_stacking'] = df_preds.mean(axis=1)
    print(f'{model_name}_predict')
    print(df_preds.head())
    return df_preds[[f'{model_name}_stacking']]


def calc_rsmle(y_pred, data):
    """LightGBMのカスタムメトリック(RMSLE)"""
    y_true = data.get_label() # lgb.Dataset() から 目的変数を取得
    metric = np.sqrt(mean_squared_error(y_true, y_pred))
    return 'rmsle', metric, False
