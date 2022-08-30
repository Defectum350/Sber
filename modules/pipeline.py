import logging
import os
from datetime import datetime
import dill
import pandas
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Fast api
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '.')


def merge_two_df(df: pandas.DataFrame, df2: pandas.DataFrame) -> pandas.DataFrame:
    df_all = df.merge(df2[['event_action', 'session_id']], on='session_id', how='outer')
    return df_all


def filter_df_f(df: pandas.DataFrame) -> pandas.DataFrame:
    df_new = df.loc[df['utm_source'].notna()]
    df_new = df_new.loc[df_new['client_id'].notna()]
    return df_new


def filter_df_t(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.loc[df['CR'].notna()]
    return df


def drop_dublicates(df: pandas.DataFrame) -> pandas.DataFrame:
    mask = df.duplicated(subset=['session_id', 'client_id', 'CR'])
    df_new = df.loc[~mask]
    return df_new


def create_feature_sc(df: pandas.DataFrame) -> pandas.DataFrame:
    def measure_device_screen(df_new):
        row = df_new['device_screen_resolution'].split('x')
        try:
            f, s = int(row[0]), int(row[1])
        except ValueError:
            f, s = 414, 896

        if f * s > 1650000:
            return 'high'
        elif 1650000 > f * s > 900000:
            return 'medium'
        else:
            return 'low'

    df['screen_category'] = df.apply(measure_device_screen, axis=1)
    return df


def create_date(df: pandas.DataFrame) -> pandas.DataFrame:
    import pandas
    df['visit_date'] = pandas.to_datetime(df['visit_date'], format="%Y/%m/%d")
    df['Year'] = df['visit_date'].dt.strftime('%Y')
    df['Month'] = df['visit_date'].dt.strftime('%m')
    df['Day'] = df['visit_date'].dt.strftime('%d')
    return df


def visit_time(df: pandas.DataFrame) -> pandas.DataFrame:
    import pandas
    df['visit_time'] = pandas.to_datetime(df['visit_time'], format="%H:%M:%S")

    def times_of_day(df_new):
        t = df_new['visit_time'].hour
        if t > 18:
            return 'Evening'
        elif 18 > t > 12:
            return 'afternoon'
        elif 12 > t > 6:
            return 'morning '
        else:
            return 'Night'

    df['times_of_day'] = df.apply(times_of_day, axis=1)
    return df


def city_category(df: pandas.DataFrame) -> pandas.DataFrame:
    import pandas
    city = pandas.read_csv(f'{path}/data/city/csvData.csv')
    huge_cities = city.query('Population >= 5000000')['Name'].tolist()
    big_cities = city.query('5000000 > Population >= 1000000')['Name'].tolist()
    medium_cities = city.query('1000000 > Population >= 500000')['Name'].tolist()

    def filtera(df_new):
        c = df_new['geo_city']
        if c in huge_cities:
            return "huge city"
        elif c in big_cities:
            return "big city"
        elif c in medium_cities:
            return "medium city"
        else:
            return "small city"

    df['city_category'] = df.apply(filtera, axis=1)
    return df


def drop_column(df: pandas.DataFrame) -> pandas.DataFrame:
    columns_to_drop = [
        'utm_keyword',
        'device_os',
        'device_model',
        'session_id',
        'client_id',
        'visit_date',
        'visit_number',
        'device_screen_resolution',
        'event_action',
        'visit_time']
    return df.drop(columns_to_drop, axis=1)


def pipeline() -> None:
    df_hit = pandas.read_csv(f'{path}/data/train/ga_hits.csv', low_memory=False)
    df_ses = pandas.read_csv(f'{path}/data/train/ga_sessions.csv', low_memory=False)
    numerical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('min_scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = Pipeline(steps=[
        ('create_date', FunctionTransformer(create_date)),
        ('create_visit_time', FunctionTransformer(visit_time)),
        ('create_sc', FunctionTransformer(create_feature_sc)),
        ('create_city_category', FunctionTransformer(city_category)),
        ('drop_column', FunctionTransformer(drop_column))])
    full_pipeline = ColumnTransformer(transformers=[
        ('num', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    df_all = merge_two_df(df_ses, df_hit)
    df_all = filter_df_f(df_all)
    mask = ['sub_car_claim_click',
            'sub_car_claim_submit_click',
            'sub_open_dialog_click',
            'sub_custom_question_submit_click',
            'sub_call_number_click',
            'sub_callback_submit_click',
            'sub_submit_success',
            'sub_car_request_submit_click']
    df_all['CR'] = df_all['event_action'].isin(mask).astype(int)
    df_all = filter_df_t(df_all)
    df_all = drop_dublicates(df_all)
    X = df_all.drop('CR', axis=1)
    y = df_all.CR
    y = y.astype(int)

    model = XGBClassifier(use_label_encoder=False,
                          objective='binary:logistic',
                          n_jobs=-1,
                          learning_rate=0.1,
                          max_depth=6,
                          scale_pos_weight=40,
                          subsample=0.7,
                          tree_method='exact',
                          gamma=3,
                          reg_alpha=15,
                          reg_lambda=15,
                          n_estimators=100,
                          max_delta_step=10,
                          num_parallel_tree=3,
                          min_child_weight=1
                          )
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('full_transform', full_pipeline),
        ('classifier', model)
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    score = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe.fit(X, y)
    print(f'XGBClassifier: , roc_auc: {score.mean():.4f}')
    model_with_metadata = {'model': pipe,
                           'metadata': {
                               'name': 'Client Sber prediction model',
                               'author': 'Sergei Molchanov',
                               'data': datetime.now(),
                               'roc_auc': score.mean(),
                               'version': 1.0
                           }}

    model_filename = f'{path}/data/models/sber_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump(model_with_metadata, file)

    logging.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
