# Module containing helper functions for stocks

# Import necessary libraries
import streamlit as st
# Data manipulation
import numpy as np
import pandas as pd
from numpy import array

from datetime import datetime

# Stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm

# Deep Learning
from tensorflow.keras.models import load_model

# Streamlit App

## Process data for time series
@st.cache_data # Cache processed dataframe
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

## Transform column for univariate time series modelling
@st.cache_data # Cache processed dataframe
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

## Train/Test Split
@st.cache_data # Cache results
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

## Define a function to extract the single element from a cell
def extract_single_element(cell_value):
    if isinstance(cell_value, np.ndarray) and len(cell_value) == 1:
        return cell_value[0]
    else:
        return cell_value

## Obtain univariate predictions (5 days ahead)
@st.cache_data # Cache results
def model_predictions_uv_5 (df, X, y, _model, n_test, test = True):
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
        single_step_pred = _model.predict(input_sequence)[:, 0, 0]
        model_preds.append(single_step_pred)

    # When reach last input of X - predict for all 5 steps
    input_sequence = X[-1:]
    multi_step_pred = _model.predict(input_sequence)
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

## Feature Engineering
@st.cache_data
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

## Prepare data for multivariate timeseries analysis
@st.cache_data
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

## Multivariate test/train split
@st.cache_data
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

## Merge initial table with predictions from uv models
@st.cache_data
def merge_uv_preds (df_clean, model1_predictions, model2_predictions, model3_predictions, model4_predictions):
    df_all = df_clean.merge(model1_predictions.add_prefix('model1_'), left_index=True, right_index=True)
    df_all = df_all.merge(model2_predictions.add_prefix('model2_'), left_index=True, right_index=True)
    df_all = df_all.merge(model3_predictions.add_prefix('model3_'), left_index=True, right_index=True)
    df_all = df_all.merge(model4_predictions.add_prefix('model4_'), left_index=True, right_index=True)
    return df_all

## Add seasonal decomposition
@st.cache_data
def season_decomp(df_clean):
    # Decomposition (day)
    decomp_days = sm.tsa.seasonal_decompose(df_clean['close'], model = 'additive')
    # add the decomposition data
    df_clean["Trend"] = decomp_days.trend
    df_clean["Seasonal"] = decomp_days.seasonal
    df_clean["Residual"] = decomp_days.resid
    return df_clean


## Load UV models
@st.cache_resource
def load_uv_models():
    model1 = load_model('stock1_model1_uv.h5')
    model2 = load_model('stock1_model2_uv.h5')
    model3 = load_model('stock1_model3_uv.h5')
    model4 = load_model('stock1_model4_uv.h5')
    return model1, model2, model3, model4

## Load Ensemble MODELS
@st.cache_resource
def load_ens_models():
    # Load models
    ens_model1 = load_model('stock1_model1_ens.h5')
    ens_model2 = load_model('stock1_model2_ens.h5')
    ens_model3 = load_model('stock1_model3_ens.h5')
    ens_model4 = load_model('stock1_model4_ens.h5')
    ens_model5 = load_model('stock1_model5_ens.h5')
    return ens_model1, ens_model2, ens_model3, ens_model4, ens_model5

## Ensemble Model Predictions:
@st.cache_data
def ens_model_predict (X_train, _ens_model):
    # Prepare for predictions
    input_4_preds = X_train[-1]
    input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
    input_4_preds = input_4_preds.reshape(input_shape)
    # Make predictions for the next 5 days
    model_predictions = _ens_model.predict(input_4_preds)

    return model_predictions
