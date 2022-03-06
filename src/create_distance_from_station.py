"""宿泊施設と各駅との距離を計算し、csvファイルを吐き出す"""

import os
os.chdir('/Users/kinoshitashouhei/Desktop/competitions/05_Prob_Space/accommodation_charge_prediction/')

import numpy as np
import pandas as pd
import warnings
from geopy.distance import geodesic
from src.config import *

warnings.filterwarnings('ignore')

LIST_USE_COL = [
    COL_ID,
    COL_HOST_ID,
    COL_LATITUDE,
    COL_LONGITUDE
]


def main():
    df_train = pd.read_csv('input/train_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_test = pd.read_csv('input/test_data.csv', parse_dates=[COL_LAST_REVIEW], dtype=DICT_DTYPES)
    df_station_list = pd.read_csv('input/station_list.csv')
    
    list_distance_columns = [f'distance_{station}' for station in list(df_station_list[COL_STATION_NAME].values)]
    
    df_all = pd.concat([df_train[LIST_USE_COL],
                        df_test[LIST_USE_COL]]).reset_index(drop=True)
    
    list_distance_from_station = get_distance(df_all, df_station_list)
    
    df_distance_from_station = pd.DataFrame(data=list_distance_from_station,
                                            columns=list_distance_columns)
    
    df_train_distance_from_station = df_distance_from_station[:df_train.shape[0]].reset_index(drop=True)
    df_test_distance_from_station = df_distance_from_station[df_train.shape[0]:].reset_index(drop=True)
    
    df_train_distance_from_station.to_csv('input/train_data_distance_from_station.csv', index=False)
    df_test_distance_from_station.to_csv('input/test_data_distance_from_station.csv', index=False)


def get_distance(df, df_station):
    list_distance = []
    for i in range(len(df)):
        list_tmp = []
        coordinate = (df.loc[i, COL_LATITUDE], df.loc[i, COL_LONGITUDE])
        for j in range(len(df_station)):
            compare_coordinate = (df_station.loc[j, COL_LATITUDE], df_station.loc[j, COL_LONGITUDE])
            list_tmp.append(geodesic(coordinate, compare_coordinate).km)
        list_distance.append(list_tmp)
    return list_distance

if __name__ == '__main__':
    main()
