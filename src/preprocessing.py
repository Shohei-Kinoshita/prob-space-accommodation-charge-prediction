import re
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
from config import *


def nearest_station(df, df_station):
    """
    2点の座標から、最寄り駅のインデックスをリストで取得
    実行時間：全データ(14,000件程度)で7min 24s
    """
    list_distance = []
    for i in range(len(df)):
        list_tmp = []
        coordinate = (df.loc[i, COL_LATITUDE], df.loc[i, COL_LONGITUDE])
        for j in range(len(df_station)):
            compare_coordinate = (df_station.loc[j, COL_LATITUDE], df_station.loc[j, COL_LONGITUDE])
            list_tmp.append(geodesic(coordinate, compare_coordinate).km)
        list_distance.append(list_tmp.index(min(list_tmp)))
    return list_distance


def remove_symbol(text):
    """
    name列の不要な記号を除去
    """
    code_regex = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％★✣♪◎☆￫〜◇✋▲△⭐︎丨❤▶☀️※《》☕️✦♯♬♡]')
    cleaned_text = text.replace(r'wi-fi', 'wifi')  # 記号除去の際の単語分割を防ぐ
    cleaned_text = code_regex.sub(' ', cleaned_text)
    return cleaned_text


def create_elapsed_days(df):
    """
    2020/4/30までの経過日数
    """
    df[COL_ELAPSED_DAYS] = (datetime.datetime(2020, 4, 30) - df[COL_LAST_REVIEW]).dt.days
    return df


def enc_categorical(df, col_list, method):
    """
    カテゴリカル変数に対して、one-hotかlabel-encを行う
    """
    if method == 'one-hot':
        df = pd.get_dummies(df, columns=col_list, drop_first=True)
        return df
    elif method == 'label-enc':
        for col in col_list:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df
