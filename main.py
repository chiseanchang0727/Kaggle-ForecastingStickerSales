import pandas as pd
import numpy as np
from src.preprocessing import create_time_features, imputation, encoding, split_and_standardization, target_transformation, reverse_target_transformation
from src.feature_engineering import generate_lag_features_by_group
from src.pred_and_eval import Predictor

def get_data():

    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    data = pd.concat([train_data, test_data], axis=0).set_index('id')
    train_data = train_data.set_index('id')
    test_data = test_data.set_index('id')

    train_idx = data.iloc[:train_data.shape[0]].index
    test_idx = data.iloc[train_data.shape[0]:].index

    return train_data, test_data

def data_transformation(df, group_cols, mode):

    df = create_time_features(df)
    if mode == 'train':
        df = imputation(df, group_cols)
        df = target_transformation(df)
        
    # df = generate_lag_features_by_group(df, group_cols)
    df = df.drop('date', axis=1)
    df = encoding(df)

    return df

def main():

    df_train, df_test = get_data()

    
    group_cols = ['country','store','product']
    df_train = data_transformation(df_train, group_cols, mode='train')
    df_test = data_transformation(df_test, group_cols, mode='test')

    # df_test = data_processing(df_test, group_cols, type='test')

    # df_train = df[df.index.isin(train_idx)]
    # df_test = df[df.index.isin(test_idx)].drop(['num_sold'], axis=1)
    scaler_train, X_train, X_valid, scaler_target, y_train, y_valid = split_and_standardization(df_train, target_col='num_sold', test_size=0.2)

    predictor = Predictor(X_train, X_valid, y_train, y_valid)
    
    mape = predictor.train_and_eval()
    # mse, y_pred = prediction(X_train_scaled, X_valid_scaled, y_train_scaled, y_valid_scaled)

    print(f"MAPE: {mape: .5f}")

    df_test_scaled = scaler_train.transform(df_test)    

    y_test_pred = predictor.predict_on_test(df_test_scaled)

    # y_test_pred_revert = np.floor(scaler_target.inverse_transform(y_test_pred.reshape(-1, 1)))

    y_test_pred_revesed = reverse_target_transformation(y_test_pred)
    
    sumbmission = pd.DataFrame({
        'id': df_test.index,
        'num_sold': y_test_pred_revesed.flatten()
    })

    sumbmission.to_csv('submission.csv', index=False)

    print('Done.')

if __name__ == '__main__':
    main()