import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model('trained_lstm_model.h5')
print("Loaded pre-trained model.")

# Load dataset
EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# Prepare data for predictions
sensor_column = 'Sensor1'
time_column = 'Time (seconds)'
scaler = MinMaxScaler()
sensor_data = EC[[sensor_column]].values
scaled_data = scaler.fit_transform(sensor_data)

# Generate future predictions
sequence_length = 50
future_time = np.linspace(EC[time_column].max() + 1, EC[time_column].max() + 100, 100).reshape(-1, 1)
future_input = scaled_data[-sequence_length:]
predictions = []

for _ in range(100):
    future_pred = model.predict(future_input.reshape(1, -1, 1))
    predictions.append(future_pred[0, 0])
    future_input = np.append(future_input[1:], future_pred, axis=0)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Enhanced Gas Sensor Dashboard with LSTM Predictions", style={'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Gas Concentrations', children=[
            dcc.Loading(
                id="loading-1",
                children=[dcc.Graph(id='gas-concentration')],
                type="circle"
            )
        ]),
        dcc.Tab(label='Sensor Correlations', children=[
            dcc.Loading(
                id="loading-2",
                children=[dcc.Graph(id='sensor-correlation')],
                type="circle"
            )
        ]),
        dcc.Tab(label='Distribution Analysis', children=[
            dcc.Loading(
                id="loading-3",
                children=[dcc.Graph(id='distribution-analysis')],
                type="circle"
            )
        ]),
        dcc.Tab(label='Peak Detection', children=[
            dcc.Loading(
                id="loading-4",
                children=[dcc.Graph(id='peak-detection')],
                type="circle"
            )
        ]),
        dcc.Tab(label='Anomaly Detection', children=[
            dcc.Loading(
                id="loading-5",
                children=[dcc.Graph(id='anomaly-detection')],
                type="circle"
            )
        ]),
        dcc.Tab(label='Trend Analysis', children=[
            dcc.Loading(
                id="loading-6",
                children=[dcc.Graph(id='trend-analysis')],
                type="circle"
            )
        ]),
        dcc.Tab(label='AI Prediction', children=[
            dcc.Loading(
                id="loading-7",
                children=[dcc.Graph(id='ai-prediction')],
                type="circle"
            )
        ])
    ])
])

# Callbacks for graphs
@app.callback(
    [Output('gas-concentration', 'figure'),
     Output('sensor-correlation', 'figure'),
     Output('distribution-analysis', 'figure'),
     Output('peak-detection', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('trend-analysis', 'figure'),
     Output('ai-prediction', 'figure')],
    [Input('gas-concentration', 'id')]
)
def update_graphs(tab_selected):
    # Gas Concentrations
    fig1 = px.line(EC, x=time_column, y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], 
                   title='Gas Concentrations Over Time')

    # Sensor Correlation Heatmap
    corr = EC[[f'Sensor{i}' for i in range(1, 17)]].corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlation Heatmap')

    # Distribution Analysis
    fig3 = px.histogram(EC, x=[f'Sensor{i}' for i in range(1, 5)], nbins=50,
                        title='Sensor Value Distribution')

    # Peak Detection
    peaks, _ = find_peaks(sensor_data.flatten(), height=sensor_data.mean() + sensor_data.std())
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
    fig4.add_trace(go.Scatter(x=EC[time_column].iloc[peaks], y=sensor_data[peaks], 
                              mode='markers', name='Peaks', marker=dict(color='red')))
    fig4.update_layout(title='Peak Detection for Sensor1')

    # Anomaly Detection
    anomalies = sensor_data.flatten() > sensor_data.mean() + 2 * sensor_data.std()
    fig5 = px.scatter(x=EC[time_column], y=sensor_data.flatten(), color=anomalies,
                      title='Anomaly Detection in Sensor1', labels={'color': 'Anomaly'})

    # Trend Analysis
    trend = pd.Series(sensor_data.flatten()).rolling(window=50).mean()
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
    fig6.add_trace(go.Scatter(x=EC[time_column], y=trend, mode='lines', name='Trend', line=dict(color='red')))
    fig6.update_layout(title='Trend Analysis for Sensor1')

    # AI Prediction
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Actual Data'))
    fig7.add_trace(go.Scatter(x=future_time.flatten(), y=predictions.flatten(), mode='lines', name='Predicted Data',
                              line=dict(color='green')))
    fig7.update_layout(title='AI Prediction for Future Data')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

