def label_enc(df, enc_col):
    """リストで指定した列をラベルエンコード"""
    for col in enc_col:
        le = LabelEncoder()
        df[col] = le.transfrom(df[col])


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
