from datetime import datetime as dt

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# model
#from model import prediction


def get_stock_price_fig(df):
    fig = go.Figure(go.Candlestick(
        x=df.Date,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close

    ))

    return fig


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
      #  "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        html.Div(
            [
                # Navigation
                #Heading of the page
                html.H1(children= "Welcome to the Stock Dash App", className="start",style={
                    'text-align' : 'center',
                    'color' : '#000000'
                }),

                #Text area to input the stock code
                html.Div([
                    html.P("Input stock code: ",style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text",placeholder='enter the stock code',required=True,style={
                            'margin' : '5px',
                            'width' : '180px',
                            'height' : '25px',
                        }),
                        html.Button("Submit", id='submit',style={
                            'margin' : '4px',
                            'color' : 'blue'
                        }),
                    ],
                             className="form")
                ],
                         className="input-place"),

                html.Br(),
                #Date-Picker-Range to pick the date
                html.P('Pick the start date', style={
                    'margin': '5px',
                    'fontSize': 20
                }),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(2000, 1, 1),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                         className="date"),

                html.Br(),
                html.Div([
                html.P('Click the "Stock Price" button to visualise the stock',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    #Button to show stock price
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock",style=
                    {
                       'margin' : '5px',
                       'color': 'blue',
                       'fontSize': 20}
                    ),

                html.Br(),
                html.Br(),
                html.P('Click the "Indicators" button to plot the EMA vs Date graph',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    #Button to indicate the
                    html.Button("Indicators",
                                className="indicators-btn",
                                id="indicators",
                                style={
                                    'margin' : '5px',
                                    'color' : 'blue',
                                    'fontSize' : 20
                                }),

                    html.Br(),
                    html.Br(),
                    html.P('Input number of days:',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder="enter the number of days",
                              required=True,
                              style={
                              'margin' : '5px',
                              'width' : '180px',
                              'height' : '25px'
                        }),

                    #Button to forecast the stock
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast",style={
                            'margin' : '4px',
                            'color' : 'blue'
                        })
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),

        # content
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo",style={
                            'margin-top' : '15px'
                        }),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content")
            ],
            className="content"),
    ],
    className="container")


# callback for company info
@app.callback([
    Output("description", "children"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])


def update_data(n, val):  # inpur parameter(s)
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "Stonks", None, None, None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            var = df[['shortName', 'longBusinessSummary']]
            return df['longBusinessSummary'].values[0], df['shortName'].values[0], None, None, None


# callback for stock graphs
@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])

#method to show the stock price
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
        #raise PreventUpdate
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]


# callback for indicators
@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])

#method to show the indicators
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])

#method to forecast
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

#method for prediction
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Your TensorFlow code here
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

def prediction(stock, n_days):
    # Load the data
    df = yf.download(stock, period='6mo')
    df.reset_index(inplace=True)

    # Prepare the data
    data = df['Close'].values
    data = data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data_len = int(np.ceil(len(scaled_data) * 0.9))

    train_data = scaled_data[0:int(train_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.1, callbacks=[early_stop])

    # Create the testing data set
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = []
    y_test = data[train_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Predict the future stock prices
    future_predictions = []
    last_60_days = test_data[-60:]
    for _ in range(n_days):
        next_pred = model.predict(last_60_days.reshape(1, 60, 1))
        future_predictions.append(next_pred[0, 0])
        last_60_days = np.append(last_60_days[1:], next_pred, axis=0)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Prepare dates for the future predictions
    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    # Plot the results
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=future_predictions,
            mode='lines+markers',
            name='Predicted Data'))
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price",
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
