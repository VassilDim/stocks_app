
###################################################################
#### Establish a requirements.txt (all the imported libraries) ####
###################################################################

## Import libraries and modules ####
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import module with functions
import importlib
import module_stocks_predict_strmlt as md
#### END


## Title and Descriptions ####
st.set_page_config (
    page_title="stock_price_predict",
    page_icon="✅",
    layout="wide"
)

# Dashboard title
st.title("Stock price prediction for the next 5 business days based on real-time data")

# Disclaimer
c1, c2, c3 = st.columns([1, 1, 1])
c1.text("DISCLAIMER:\n This app is for training purposes ONLY!\n Information depicted here is NOT intended as any\n form of financial advise.")
c3.text("Author: Vassil Dimitrov\nLast updated: 2023-09-25\nLinkedIn: https://www.linkedin.com/in/vassildim/\nGitHub: https://github.com/VassilDim/VassilDim")
#### END

#############################
#### Choose stock (type) ####
#############################


## Load data directly ####
# Define dataset url
dataseturl = 'https://stooq.com/q/d/l/?s=fcel.us&i=d'
# Read table directly from a URL
@st.cache_data # If cashed, leave
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataseturl)
# Load data from site
df = get_data()
# Clean up data
df_clean = md.process_stock_table(df)
df_historical = df_clean.copy()

# Decomposition (day)
decomp_days = sm.tsa.seasonal_decompose(df_clean['close'], model = 'additive')
# add the decomposition data
df_clean["Trend"] = decomp_days.trend
df_clean["Seasonal"] = decomp_days.seasonal
df_clean["Residual"] = decomp_days.resid
#### END


## Prep table for modelling ####

# Load univariate models
model1 = load_model('stock1_model1_uv.h5')
model2 = load_model('stock1_model2_uv.h5')
model3 = load_model('stock1_model3_uv.h5')
model4 = load_model('stock1_model4_uv.h5')

# Obtain univariate predictions:
n_test=0
# Obtain input sequences MODEL1 univariate
X, y = md.prep_4_time_series_uv (df_clean['close'], element_history=7, to_predict=5)
X_train, y_train = md.train_test_split_transform_uv_LSTM (X, y, n_test)
# Obtain predictions MODEL1
model1_predictions = md.model_predictions_uv_5(df_clean, X_train, y_train, model1, n_test=0, test=False)

# Obtain input sequences MODEL2 uv
X, y = md.prep_4_time_series_uv (df_clean['close'], element_history=2, to_predict=5)
X_train, y_train = md.train_test_split_transform_uv_LSTM(X, y, n_test)
# Obtain predictions MODEL2
model2_predictions = md.model_predictions_uv_5(df_clean, X_train, y_train, model2, n_test=0, test=False)

# Obtain input sequences MODEL3 uv
X, y = md.prep_4_time_series_uv (df_clean['close'], element_history=3, to_predict=5)
X_train, y_train = md.train_test_split_transform_uv_LSTM (X, y, n_test)
# Obtain predictions MODEL3
model3_predictions = md.model_predictions_uv_5(df_clean, X_train, y_train, model3, n_test=0, test=False)

# Obtain input sequences MODEL4 uv
X, y = md.prep_4_time_series_uv (df_clean['close'], element_history=5, to_predict=5)
X_train, y_train = md.train_test_split_transform_uv_LSTM (X, y, n_test)
# Obtain predictions MODEL4
model4_predictions = md.model_predictions_uv_5(df_clean, X_train, y_train, model4, n_test=0, test=False)

# Update table:
df_all = df_clean.merge(model1_predictions.add_prefix('model1_'),
                  left_index=True, right_index=True)
df_all = df_all.merge(model2_predictions.add_prefix('model2_'),
                  left_index=True, right_index=True)
df_all = df_all.merge(model3_predictions.add_prefix('model3_'),
                  left_index=True, right_index=True)
df_all = df_all.merge(model4_predictions.add_prefix('model4_'),
                  left_index=True, right_index=True)

# Feature engineering and cleanup
df = md.add_daily_pc_volume_pc(df_all)
df_clean = df.copy()
df_all = df.copy()
df_all.drop(columns = ['open', 'volume', 'high', 'low', 'Trend', 'Seasonal', 'Residual'], inplace=True)

# Normalize values
# Initialize a standard scaler
scaler = StandardScaler()
# Scale
df_all_scaled = scaler.fit_transform (df_all)
#### END

## MAKE PREDICTIONS WITH ENSEMBLE MODELS ####
# Load models
ens_model1 = load_model('stock1_model1_ens.h5')
ens_model2 = load_model('stock1_model2_ens.h5')
ens_model3 = load_model('stock1_model3_ens.h5')
ens_model4 = load_model('stock1_model4_ens.h5')
ens_model5 = load_model('stock1_model5_ens.h5')

# Make predictions
# Obtain Timesteps MODEL 1 ens
X, y = md.prep_4_time_series (df_all, df_all_scaled, element_history=4)
# Do not split into test and train
X_train, y_train = md.train_test_split_transform_LSTM (X, y, n_test=0)
# Prepare for predictions
input_4_preds = X_train[-1]
input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
input_4_preds = input_4_preds.reshape(input_shape)
# Make predictions for the next 5 days
model1_predictions = ens_model1.predict(input_4_preds)

# Obtain Timesteps MODEL 2 ens
X, y = md.prep_4_time_series (df_all, df_all_scaled, element_history=2)
# Split into test (100) and train
X_train, y_train = md.train_test_split_transform_LSTM (X, y, n_test=0)
# Prepare for predictions
input_4_preds = X_train[-1]
input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
input_4_preds = input_4_preds.reshape(input_shape)
# Make predictions for the next 5 days
model2_predictions = ens_model2.predict(input_4_preds)

# Obtain Timesteps MODEL 3 ens
X, y = md.prep_4_time_series (df_all, df_all_scaled, element_history=3)
# Split into test (100) and train
X_train, y_train = md.train_test_split_transform_LSTM (X, y, n_test=0)
# Prepare for predictions
input_4_preds = X_train[-1]
input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
input_4_preds = input_4_preds.reshape(input_shape)
# Make predictions for the next 5 days
model3_predictions = ens_model3.predict(input_4_preds)

# Obtain Timesteps MODEL 4 ens
X, y = md.prep_4_time_series (df_all, df_all_scaled, element_history=4)
# Split into test (100) and train
X_train, y_train = md.train_test_split_transform_LSTM (X, y, n_test=0)
# Prepare for predictions
input_4_preds = X_train[-1]
input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
input_4_preds = input_4_preds.reshape(input_shape)
# Make predictions for the next 5 days
model4_predictions = ens_model4.predict(input_4_preds)

# Obtain Timesteps MODEL 5 ens
X, y = md.prep_4_time_series (df_all, df_all_scaled, element_history=16)
# Split into test (100) and train
X_train, y_train = md.train_test_split_transform_LSTM (X, y, n_test=0)
# Prepare for predictions
input_4_preds = X_train[-1]
input_shape = (1, X_train[-1].shape[0], X_train[-1].shape[1])
input_4_preds = input_4_preds.reshape(input_shape)
# Make predictions for the next 5 days
model5_predictions = ens_model5.predict(input_4_preds)
#### END


## Package Predictions into a Table ####
# Obtain a list of indexes (next 5 business days since last in df)
last_available_date = df_all.index[-1]
next_business_day = last_available_date + pd.tseries.offsets.BDay()
pred_indexes = pd.bdate_range(
    start=next_business_day, periods=5, freq='B'
).to_pydatetime().tolist()
# Package into a table
ensemble_models_predictions = pd.DataFrame({'model1' : model1_predictions.flatten(),
                                            'model2' : model2_predictions.flatten(),
                                            'model3' : model3_predictions.flatten(),
                                            'model4' : model4_predictions.flatten(),
                                            'model5' : model5_predictions.flatten()},
                                           index = pred_indexes
                                          )
#### END


## Plot historical data values WITH PREDICTIONS ####

st.subheader('Predictions from 5 LSTM Ensemble Models and Historical Data')

# Define column selection for plotting
selected_columns = st.multiselect("Select values to plot", df_historical.columns[:-1])

# Create a plot
if selected_columns: # Check that at least 1 selected

    # Create Streamlit widgets for y-axis range
    y_min = st.number_input("y-min",\
    min_value=df_historical[selected_columns].min().min(),\
    max_value=df_historical[selected_columns].max().max(),\
    value=df_historical[selected_columns].min().min())
    y_max = st.number_input("y-max",\
    min_value=df_historical[selected_columns].min().min(),\
    max_value=df_historical[selected_columns].max().max(),\
    value=df_historical[selected_columns].max().max())

    # Create plotly with multiple lines for each figure
    fig = go.Figure()
    # Define line colours:
    line_colors = ['blue', 'green', 'red', 'orange']
    line_colors2 = ['purple', 'blueviolet', 'magenta', 'chocolate', 'crimson']
    for i, col in enumerate(selected_columns):
        line_color = line_colors[i]
        fig.add_trace(go.Scatter(
        x=df_historical.index,
        y=df_historical[col],
        name=col,
        line=dict(color=line_color, dash='dot'),
        opacity=0.5))
    for j in range(5):
        line_color2 = line_colors2[j]
        fig.add_trace(go.Scatter(
        x=ensemble_models_predictions.index,
        y=ensemble_models_predictions.iloc[:,j],
        mode = 'lines+markers',
        marker = dict(color=line_color2, symbol='circle'),
        line=dict(color=line_color2),
        name = 'model_'+str(j+1),
        opacity=0.5
    ))

    # Update layout
    fig.update_layout(
    yaxis_title='Value',
    xaxis_title='Date',
    title='Daily Value in USD'
    )

    # Update y-axis range based on user input
    fig.update_yaxes(range=[y_min, y_max])
    # Set the x-axis range to cover only the relevant date range
    fig.update_xaxes(range=[df_historical.index.min(), ensemble_models_predictions.index.max()])
    # Activate X-axis rangeslider
    fig.update_xaxes(rangeslider_visible=True)

    # Show in app
    st.plotly_chart(fig, use_container_width=True)

else: # Prompt for a selection
    st.warning('Please select at least 1 category.')

# Define space to take
col1, col2 = st.columns([2, 1])
# Create a plot for volume traded
# Create Streamlit widgets for y-axis range
with col1:
    y_min_volume = st.number_input("y-min", min_value=int(df_historical['volume'].min()),\
    max_value=int(df_historical['volume'].max()),\
    value=int(df_historical['volume'].min()))
    y_max_volume = st.number_input("y-max",\
    min_value=int(df_historical['volume'].min()),\
    max_value=int(df_historical['volume'].max()),\
    value=int(df_historical['volume'].max()))

    # Create a plot
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Scatter(
            x=df_historical.index,
            y=df_historical['volume'],
            line=dict(color='aquamarine', dash='dot')))
    # Update layout
    fig_volume.update_layout(
    yaxis_title='Volume traded',
    xaxis_title='Date',
    title='Daily Value of Volume Traded'
    )
    # Update y-axis range based on user input
    fig_volume.update_yaxes(range=[y_min_volume, y_max_volume])
    # Activate X-axis rangeslider
    fig_volume.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_volume, use_container_width=True)
# Plot table with predictions
with col2:
    ensemble_models_predictions.style.set_caption('Predictions for next 5 days\n(5 models)')
    st.dataframe(ensemble_models_predictions)
#### END

## Trend and seasonal decomposition graphs (choose between trend, seasonality and residuals) ####
cols = ["Trend", "Seasonal", "Residual"]
colors = ['turquoise', 'pink', 'red']

st.subheader('Diagnostic Plots For Time Series Decomposition for Value at Close')

# Create Streamlit columns layout
col1, col2, col3 = st.columns(3)

# Create and display each plot side-by-side horizontally
for i, col in enumerate(cols):
    figa = go.Figure(go.Scatter(x=df_clean.index,
                                y=df_clean[col],
                                name=col,
                                line=dict(color=colors[i])
                               ))
    # Update the layout of each figure to add a range slider
    figa.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'), title=col)
    # Display figure in appropriate column
    if i==0:
        col1.plotly_chart(figa, use_container_width=True)
    elif i==1:
        col2.plotly_chart(figa, use_container_width=True)
    elif i==2:
        col3.plotly_chart(figa, use_container_width=True)
#### END
