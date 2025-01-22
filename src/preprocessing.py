import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import holidays

def create_sinusoidal_transformation_year_month_day(df, col_name, year, month, day, period):
    """
    Adds sinusoidal transformation columns (sin and cos) for year, month, day.
    """
    # df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[year] * df[month] * df[day] / period)
    # df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[year] * df[month] * df[day] / period)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    return df

def create_time_features(df: pd.DataFrame, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Time-based features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofWeek'] = df[date_col].dt.dayofweek
    # df['weekend'] = np.where(df['dayofWeek']>5, 1, 0)

    # for country in df.country.unique():
    #     holiday_cal = holidays.CountryHoliday(country=country)
    #     df[f'{country}_holiday'] = df[date_col].apply(lambda x: x in holiday_cal).astype(int)

    # df = create_sinusoidal_transformation_year_month_day(df, 'date', "year", "month", "day", 12)

    return df


def imputation(df: pd.DataFrame, group_by: list):


    # df.loc[df.index.isin(train_idx), 'num_sold'] = df.loc[df.index.isin(train_idx), 'num_sold'].fillna(0)

    df['num_sold'] = df['num_sold'].fillna(df['num_sold'].median())

    # df_temp = df.groupby(group_by)['num_sold'].mean().reset_index(name='avg_sold').round(0)
    # df_merge = pd.merge(df, df_temp, how='left', on=group_by)
    # df_merge['num_sold'] = np.where(df_merge['num_sold'].isna(), df_merge['avg_sold'], df_merge['num_sold'])

    # df_merge['num_sold'] = np.log1p(df_merge['num_sold'])

    return df


def encoding(df: pd.DataFrame):

    categorical_col = df.select_dtypes('object').columns.to_list()
    df_encoded = pd.get_dummies(df, columns=categorical_col, drop_first=False, dtype=int)

    return df_encoded


def data_splitting(df, target_col, test_size=0.2):

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=42)



def split_and_standardization(df, target_col, test_size):

    X_train, X_valid, y_train, y_valid = data_splitting(df, target_col, test_size)

    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)
    X_valid_scaled = scaler_train.transform(X_valid)

    scaler_target = StandardScaler()
    y_train_scaled = scaler_target.fit_transform(pd.DataFrame(y_train))
    y_valid_scaled = scaler_target.transform(pd.DataFrame(y_valid))

    return scaler_train, X_train_scaled, X_valid_scaled, scaler_target, y_train_scaled, y_valid_scaled

