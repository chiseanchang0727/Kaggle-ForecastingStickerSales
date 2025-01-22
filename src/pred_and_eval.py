import pandas as pd
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

def xgb_model():

    return XGBRFRegressor(
        n_estimators=100,        # Number of trees
        max_depth=15,             # Maximum depth of each tree
        learning_rate=0.1,       # Learning rate (shrinkage factor)
        subsample=0.8,           # Subsample ratio for training instances
        colsample_bynode=0.8,    # Subsample ratio for columns at each tree node
        random_state=42          # For reproducibility
    )

def prediction(X_train, X_valid, y_train, y_valid):

    model = xgb_model()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    
    mse = mean_absolute_percentage_error(y_valid, y_pred)


    return mse, y_pred


class Predictor:
    def __init__(self, X_train, X_valid, y_train, y_valid):
        self.model = self.get_model()
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def xgb_model():

        return XGBRFRegressor(
            n_estimators=120,        # Number of trees
            max_depth=15,             # Maximum depth of each tree
            learning_rate=0.01,       # Learning rate (shrinkage factor)
            subsample=0.8,           # Subsample ratio for training instances
            colsample_bynode=0.8,    # Subsample ratio for columns at each tree node
            random_state=42          # For reproducibility
        )


    def get_model(self):

        model = xgb_model()

        return model
    

    def train_and_eval(self):
        
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_valid)

        mape = mean_absolute_percentage_error(self.y_valid, y_pred)

        return mape

    def predict_on_test(self, df_test: pd.DataFrame):

        y_pred = self.model.predict(df_test)

        return y_pred

