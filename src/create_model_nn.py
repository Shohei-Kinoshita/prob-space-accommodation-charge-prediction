import sys
import random
import os
import numpy as np
import pandas as pd
import warnings
import gc
from tqdm import tqdm
from numpy.ma.core import argsort

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from config import *
import preprocessing as pr
import train_predict as train_predict

warnings.filterwarnings('ignore')

# ハイパーパラメータ
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 70
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 11
EARLY_STOP = True
TARGET_SIZE = 1
HIDDEN_SIZE_1 = 512
HIDDEN_SIZE_2 = 256

def main():
    args = sys.argv
    model_name = args[1]
    enc_type = args[2]
    model_save = args[3]

    print(f'model_name: {model_name}')
    print(f'enc_type: {enc_type}')
    print(f'model_save: {model_save}')

    # 乱数シードの固定
    seed_everything(seed=42)
    
    # データの読み込み
    print('read data...')
    df_train = pd.read_csv('input/train_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_test = pd.read_csv('input/test_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_train_station_info = pd.read_csv('input/train_data_distance_from_station.csv', dtype=DICT_DTYPES)
    df_test_station_info = pd.read_csv('input/test_data_distance_from_station.csv', dtype=DICT_DTYPES)
    df_train_name_features = pd.read_csv('input/train_data_name_features.csv')
    df_test_name_features = pd.read_csv('input/test_data_name_features.csv')

    df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

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
    
    # 学習、検証、モデルの保存
    num_features= X.shape[1]
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=0)
    preds = []
    va_idxs = []
    for i, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        model = Model(num_features, TARGET_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        print(f'fold: {i}')
        print('='*50)
        X_train, X_valid = X.loc[train_idx, :], X.loc[valid_idx, :]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
        train_dataset = TrainDataset(X_train, y_train)
        valid_dataset = TrainDataset(X_valid, y_valid)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

        for epoch in range(EPOCHS):
            print(f'epoch: {epoch}')
            train_loss, train_rmsle = train_fn(train_loader, model, criterion, optimizer, DEVICE)
            valid_loss, valid_rmsle, valid_preds = valid_fn(valid_loader, model, criterion, DEVICE)
            print(f'train_loss: {train_loss:.4f}, train_rmsle: {train_rmsle:.4f}')
            print(f'valid_loss: {valid_loss:.4f}, valid_rmsle: {valid_rmsle:.4f}')
            print('-'*50)
        
        va_idxs.append(valid_idx)
        preds.append(valid_preds)
    
        if model_save == 'True':
            print('='*50)
            print(f'save model: models/{model_name}_fold{i}_.pth')
            print('='*50)
            torch.save(model.state_dict(), f"models/{model_name}_fold{i}_.pth")

        del model, criterion, optimizer
        gc.collect()

    preds = np.concatenate(preds)
    va_idxs = np.concatenate(va_idxs)
    order = argsort(va_idxs)
    df_oof = pd.DataFrame(preds[order], columns=[f'{model_name}_stacking'])
    print('df_oof')
    print(df_oof.head())

    if model_save == 'True':
        list_preds_tmp = []
        inference_dataset = TestDataset(X_inference)
        inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for i in range(1, 6):
            model = Model(num_features, TARGET_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
            model.load_state_dict(torch.load(f"models/{model_name}_fold{i}_.pth"))
            model.to(DEVICE)
            predictions = inference_fn(model, inference_loader, DEVICE)
            list_preds_tmp.append(predictions)
        df_preds = pd.DataFrame({'model_1': np.squeeze(list_preds_tmp[0]),
                                 'model_2': np.squeeze(list_preds_tmp[1]),
                                 'model_3': np.squeeze(list_preds_tmp[2]),
                                 'model_4': np.squeeze(list_preds_tmp[3]),
                                 'model_5': np.squeeze(list_preds_tmp[4])})
        df_preds[f'{model_name}_stacking'] = df_preds.mean(axis=1)
        print(f'{model_name}_predict')
        print(df_preds.head())
        
        # out of foldの予測値を出力(スタッキングの特徴量として使用)
        df_oof.to_csv(f'input/train_{model_name}_out_of_fold.csv', index=False)
        df_preds[[f'{model_name}_stacking']].to_csv(f'input/test_{model_name}_out_of_fold.csv', index=False)
    print('finish')


class TrainDataset(Dataset):
    def __init__(self, features, target):
        super().__init__()
        self.features = features.reset_index(drop=True)
        self.target = target.reset_index(drop=True)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.features.loc[idx, :].values, dtype=torch.float)
        y = torch.tensor(self.target[idx], dtype=torch.float)
        return x, y


class TestDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features.reset_index(drop=True)
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.features.loc[idx, :].values, dtype=torch.float)
        return x


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size_1, hidden_size_2):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size_1))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size_1)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size_1, hidden_size_2))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size_2, hidden_size_2))
        
        self.batch_norm4 = nn.BatchNorm1d(hidden_size_2)
        self.dropout4 = nn.Dropout(0.25)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size_2, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.relu(self.dense3(x))
        
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.dense4(x)
        
        return x


def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_rmsle = 0.0

    for features, labels in tqdm(train_loader):
        optimizer.zero_grad()  # 勾配情報を初期化
        features = features.to(device)  # GPU上に転送
        labels = labels.to(device)  # GPU上に転送
        outputs = model(features)  # GPU上に転送
        outputs = torch.squeeze(outputs)  # 1次元に変換
        loss = torch.sqrt(criterion(outputs, labels))  # 損失を計算
        preds = outputs.detach().cpu().numpy()  # numpyの配列に変換
        labels = labels.detach().cpu().numpy()  # numpyの配列に変換

        train_loss += loss.item()  # 値だけ取得
        train_rmsle += np.sqrt(mean_squared_error(labels, preds))  # 評価指標の計算
        loss.backward()  # 勾配を計算
        optimizer.step()  # 勾配を更新

    train_loss /= len(train_loader)  # 1epoch分の平均を計算
    train_rmsle /= len(train_loader)  # 1epoch分の平均を計算

    return train_loss, train_rmsle


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_rmsle = 0.0
    valid_preds = []

    for features, labels in tqdm(valid_loader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(features)
            outputs = torch.squeeze(outputs)
            loss = torch.sqrt(criterion(outputs, labels))
            preds = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

        valid_loss += loss.item()
        valid_rmsle += np.sqrt(mean_squared_error(labels, preds))
        valid_preds.append(preds)

    valid_loss /= len(valid_loader)
    valid_rmsle /= len(valid_loader)
    valid_preds = np.concatenate(valid_preds)

    return valid_loss, valid_rmsle, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for features in dataloader:
        features = features.to(device)

        with torch.no_grad():
            outputs = model(features)
        
        preds.append(outputs.detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
