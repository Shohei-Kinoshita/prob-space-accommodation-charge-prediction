"""列名の定義"""
COL_ID = 'id'
COL_NAME = 'name'
COL_HOST_ID = 'host_id'
COL_NEIGHBOURHOOD = 'neighbourhood'
COL_LATITUDE = 'latitude'
COL_LONGITUDE = 'longitude'
COL_ROOM_TYPE = 'room_type'
COL_MINIMUM_NIGHTS = 'minimum_nights'
COL_NUMBER_OF_REVIEWS = 'number_of_reviews'
COL_LAST_REVIEW = 'last_review'
COL_REVIEWS_PER_MONTH = 'reviews_per_month'
COL_AVAILABILITY_365 = 'availability_365'
COL_Y = 'y'
COL_LOG_Y = 'log_y'
COL_STATION_NAME = 'station_name'
COL_CLEAN_NAME = 'clean_name'
COL_ELAPSED_DAYS = 'elapsed_days'

"""データ読み込み時の型指定"""
DICT_DTYPES = {
    COL_ID: str,
    COL_HOST_ID: str
}

"""LightGBMのハイパーパラメータ"""
DICT_PARAMS_LGB = {
      # TODO: optunaで最適化
      'task': 'train',
      'boosting_type': 'gbdt',
      'objective': 'regression',
      'metric': 'mean_squared_error',
      'learning_rate': 0.1
}

"""モデルに使用する列"""
LIST_USE_COL = [COL_NEIGHBOURHOOD,
                COL_LATITUDE,
                COL_LONGITUDE,
                COL_ROOM_TYPE,
                COL_MINIMUM_NIGHTS,
                COL_NUMBER_OF_REVIEWS,
                COL_ELAPSED_DAYS,
                COL_REVIEWS_PER_MONTH,
                COL_AVAILABILITY_365]

"""エンコードを行う列"""
LIST_ENC_COL = [COL_NEIGHBOURHOOD,
                COL_ROOM_TYPE]
