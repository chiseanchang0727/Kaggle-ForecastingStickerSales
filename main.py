import pandas as pd
import numpy as np
from src.preprocessing import create_time_features, imputation, encoding, split_and_standardization
from src.feature_engineering import generate_lag_features_by_group
from src.pred_and_eval import Predictor

def get_data():

    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    data = pd.concat([train_data, test_data], axis=0).set_index('id')

    train_idx = data.iloc[:train_data.shape[0]].index
    test_idx = data.iloc[train_data.shape[0]:].index

    return train_data, test_data

def data_processing(df, group_cols):

    # df = create_time_features(df)
    df = imputation(df, group_cols)
    # df = generate_lag_features_by_group(df, group_cols)

    df = df.drop('date', axis=1)
    df = encoding(df)

    return df

def main():

    df_train, df_test = get_data()

    
    group_cols = ['country','store','product']
    df_train = data_processing(df_train, group_cols)

    # df_test = data_processing(df_test, group_cols, type='test')

    # df_train = df[df.index.isin(train_idx)]
    # df_test = df[df.index.isin(test_idx)].drop(['num_sold'], axis=1)
    scaler_train, X_train_scaled, X_valid_scaled, scaler_target, y_train_scaled, y_valid_scaled = split_and_standardization(df_train, target_col='num_sold', test_size=0.2)

    predictor = Predictor(X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled)

    
    mape = predictor.train_and_eval()
    # mse, y_pred = prediction(X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled)

    print(f"MAPE: {mape: .5f}")

    # df_test_scaled = scaler_train.transform(df_test)    

    # y_test_pred = predictor.predict_on_test(df_test_scaled)

    # y_test_pred_revert = np.floor(scaler_target.inverse_transform(y_test_pred.reshape(-1, 1)))

    # sumbmission = pd.DataFrame({
    #     'id': df_test.index,
    #     'num_sold': y_test_pred_revert.flatten()
    # })

    # sumbmission.to_csv('submission.csv', index=False)

    print(1)

if __name__ == '__main__':
    main()