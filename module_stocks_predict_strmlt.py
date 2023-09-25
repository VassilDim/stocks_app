# Module containing helper functions for stocks

# Import necessary libraries
# Data manipulation
import numpy as np
import pandas as pd
from numpy import array

from datetime import datetime

# Stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Streamlit App
def process_stock_table (df, specific_date='2020-01-01'):

    '''
    Loads a table specified by 'path_to_csv' as string.
    Subsets the dataframe starting at 'specific_date'.
    Removes duplicated rows.
    Adds missed business days.
    Forward fills null values.
    Returns a clean dataframe only containing values for 'open', 'close' and 'volume'.
    Sets the 'date' as index
    '''

    # Convert to proper date
    target_date = pd.to_datetime(specific_date, format="%Y-%m-%d")

    # Tidy up data
    df.columns = df.columns.str.lower()
    df.rename(columns = lambda x: x.strip(), inplace=True)

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicated rows
    df.drop_duplicates(inplace=True)

    # Set index as date
    df.set_index('date', inplace=True)

    # Re-index with all business days
    first_day = df.index.min()
    last_day = df.index.max()
    full_range = pd.date_range (start=first_day, end=last_day, freq='B')
    df_clean = df.reindex(full_range)

    # Forward fill to impute missing values
    df_clean.fillna(method='ffill', inplace=True)

    # Subset for date
    df_final = df_clean.loc[(df_clean.index >= target_date),['open', 'close', 'high', 'low', 'volume']]

    return df_final

def prep_4_time_series_uv (dat_list, element_history=3, to_predict=5):

    '''
    This function takes a list (feature) 'dat_list' and creates a list of list where
    each element of the inner list 'element_history' is a value for a time step.
    The function returns the values (time steps) 'X' to be used for fitting the model
    and the target (the following time steps) 'y'.
    '''

    # Define list to hold feature vectors (time window)
    X = list()
    # Define empty list to hold response variable (next after window)
    y = list()

    dat_list = list(dat_list)

    for i in range(len(dat_list) - element_history - to_predict + 1):
        X.append(dat_list[i :i + element_history])
        y.append(dat_list[i + element_history : i + element_history + to_predict])

    # Return X and y
    return X, y

def train_test_split_transform_uv_LSTM (X_tot, y_tot, n_test):

    if n_test > 0:

        # obtain subsets for test and train
        X_train = X_tot[:-n_test]
        y_train = y_tot[:-n_test]
        X_test = X_tot[-n_test:]
        y_test = y_tot[-n_test:]

        # transform subsets for Deep NNs (CNN or LSTM)
        X = np.array(X_train)
        y = array(y_train)
        X_test = np.array(X_test)

        # reshape for 1 feature to [samples, subsequences, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        X_test = array(X_test)
        x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return X, x_test, y_train, y_test

    else:
        X = np.array(X_tot)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = np.array(y_tot)
        return X, y

# Define a function to extract the single element from a cell
def extract_single_element(cell_value):
    if isinstance(cell_value, np.ndarray) and len(cell_value) == 1:
        return cell_value[0]
    else:
        return cell_value

def model_predictions_uv_5 (df, X, y, model, n_test, test = True):
    # Initialize empty lists
    model_preds = []
    dates = []
    element_history = X.shape[1]

    # Check whether the prediction of for the test or train set
    if test:
        temp_df = df[-n_test : ] # get test portion of df
    elif not test and n_test == 0: # no test subset (predict for all data)
        temp_df = df
    else:
        temp_df = df[: -n_test] # get train portion of df

    # Obtain dates:
    pred_date = temp_df.index[element_history].date()
    dates = pd.bdate_range(start=pred_date, periods=len(X)+5, freq='B').to_pydatetime().tolist()


    # Iterate over each input timestep:
    for i in range(len(X)):
        #print('predicting date', dates[i], '\n')
        # Obtain timestep for input
        input_sequence = X[i : i+1]
        # Only keep first step of prediction
        single_step_pred = model.predict(input_sequence)[:, 0, 0]
        model_preds.append(single_step_pred)

    # When reach last input of X - predict for all 5 steps
    input_sequence = X[-1:]
    multi_step_pred = model.predict(input_sequence)
    # Add predictions
    model_preds.extend(multi_step_pred)

    # Flatten predictions
    model_preds_flat = [val for sublist in model_preds for val in sublist]

    # Construct a dataframe
    model_pred_df = pd.DataFrame({
    'predictions' : model_preds_flat
    }, index = dates)

    # Extract values from lists
    model_pred_df = model_pred_df.applymap(extract_single_element)

    return model_pred_df

def add_daily_pc_volume_pc (df):
    '''
    Adds a new column 'daily_pc' calculating the percent change between
    open and close values.
    Adds a new column 'volume_pc' calculating the percent change of the
    volume traded the current day relative to the previous day.
    '''
    # Add daily percent change
    df['daily_pc'] = (df['close'] - df['open'])/df['open'] * 100

    # Add volume change
    df['volume_pc'] = df['volume'].pct_change() * 100

    # Replace null values with 0
    df.fillna(0, inplace=True)

    return df

def prep_4_time_series (df, nparray_scaled, element_history=3, to_predict=5):
        '''
        This function takes an np.array of normalized feature (nparray_scaled)
        and a standard dataframe used for normalization (df) to extract the response
        variable (close).
        It then creates a list where each element is an array of features (columns)
        for multiple timesteps (rows).
        The function returns the values (time steps) 'X' to be used for fitting the model
        and the target (the following time steps for close only) 'y'.
        '''
        X = list()
        y = list()

        for i in range(len(df) - element_history - to_predict + 1):
            X.append (nparray_scaled [i : i + element_history])
            y.append (df['close'][i + element_history : i + element_history + to_predict])

        X = np.array(X)

        return X, y

def train_test_split_transform_LSTM (X_tot, y_tot, n_test):

    if n_test > 0:

        # obtain subsets for test and train
        X_train = X_tot[:-n_test]
        y_train = y_tot[:-n_test]
        X_test = X_tot[-n_test:]
        y_test = y_tot[-n_test:]

        # transform subsets for Deep NNs (CNN or LSTM)
        X = np.array(X_train)
        y = array(y_train)
        X_test = np.array(X_test)

        # reshape for 1 feature to [samples, subsequences, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        X_test = array(X_test)
        x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        return X, x_test, y_train, y_test

    else:
        X = np.array(X_tot)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        y = np.array(y_tot)
        return X, y
