# import dash
# from dash import dcc, html, Input, Output
# import plotly.graph_objs as go
# import numpy as np
# from scipy.signal import find_peaks

# # Step 1: Simulated GC Data
# np.random.seed(42)
# time = np.linspace(0, 100, 1000)
# peak_positions = [20, 50, 80]
# peak_amplitudes = [1, 1.5, 1.2]
# normal_peaks = sum(
#     amp * np.exp(-0.5 * ((time - pos) ** 2) / 1) for pos, amp in zip(peak_positions, peak_amplitudes)
# )
# shifted_peaks = sum(
#     amp * np.exp(-0.5 * ((time - (pos + np.random.uniform(1, 5))) ** 2) / 1)
#     for pos, amp in zip(peak_positions, peak_amplitudes)
# )
# noise = np.random.normal(0, 0.05, size=time.shape)
# signal = normal_peaks + shifted_peaks + noise

# # Step 2: Peak Detection
# peaks, _ = find_peaks(signal, height=0.1, distance=50)

# # Step 3: Dash App
# app = dash.Dash(__name__)
# app.layout = html.Div([
#     html.H1("GC Data Analysis with AI", style={"textAlign": "center"}),
#     dcc.Graph(id="gc-plot"),
#     html.Div([
#         html.Label("Noise Threshold:"),
#         dcc.Slider(
#             id="noise-threshold",
#             min=0,
#             max=0.2,
#             step=0.01,
#             value=0.1,
#             marks={i: str(round(i, 2)) for i in np.arange(0, 0.21, 0.05)}
#         )
#     ], style={"padding": "20px"})
# ])

# @app.callback(
#     Output("gc-plot", "figure"),
#     [Input("noise-threshold", "value")]
# )
# def update_graph(threshold):
#     filtered_peaks, _ = find_peaks(signal, height=threshold, distance=50)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time, y=signal, mode="lines", name="Signal", line=dict(color="blue")))
#     fig.add_trace(go.Scatter(
#         x=time[filtered_peaks], y=signal[filtered_peaks],
#         mode="markers", name="Detected Peaks", marker=dict(color="red", size=10)
#     ))

#     fig.update_layout(
#         title="Simulated GC Data Analysis",
#         xaxis_title="Time",
#         yaxis_title="Intensity",
#         template="plotly_white"
#     )
#     return fig

# if __name__ == "__main__":
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# # إعداد البيانات (محاكاة للبيانات الحقيقية)
# np.random.seed(42)
# time = np.linspace(0, 100, 1000)
# sensors = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"]
# data = {sensor: np.sin(0.1 * time + i) + 0.1 * np.random.randn(len(time)) for i, sensor in enumerate(sensors)}
# data["Time"] = time
# df = pd.DataFrame(data)

# # إنشاء تطبيق Dash
# app = dash.Dash(__name__)

# # تصميم واجهة المستخدم
# app.layout = html.Div([
#     html.H1("Gas Sensor Analysis Dashboard", style={"text-align": "center"}),

#     # اختيار الحساسات
#     html.Div([
#         html.Label("Select Sensors:"),
#         dcc.Checklist(
#             id="sensor-selector",
#             options=[{"label": sensor, "value": sensor} for sensor in sensors],
#             value=["Sensor_1", "Sensor_2"],
#             inline=True
#         )
#     ], style={"margin-bottom": "20px"}),

#     # الرسوم البيانية
#     html.Div([
#         dcc.Graph(id="time-series-plot"),
#         dcc.Graph(id="peak-distribution")
#     ])
# ])

# # الوظائف التفاعلية
# @app.callback(
#     [Output("time-series-plot", "figure"),
#      Output("peak-distribution", "figure")],
#     [Input("sensor-selector", "value")]
# )
# def update_graphs(selected_sensors):
#     # الرسم البياني للسلاسل الزمنية
#     fig1 = go.Figure()
#     for sensor in selected_sensors:
#         fig1.add_trace(go.Scatter(x=df["Time"], y=df[sensor], mode="lines", name=sensor))
#     fig1.update_layout(title="Gas Sensor Signals Over Time",
#                        xaxis_title="Time",
#                        yaxis_title="Signal Intensity")

#     # الرسم البياني لتوزيع القمم
#     peak_counts = {sensor: len(np.where(df[sensor] > df[sensor].mean() + 0.5)[0]) for sensor in selected_sensors}
#     fig2 = px.bar(x=list(peak_counts.keys()), y=list(peak_counts.values()),
#                   labels={"x": "Sensor", "y": "Number of Peaks"},
#                   title="Peak Distribution Across Sensors")
#     return fig1, fig2


# # تشغيل التطبيق
# if __name__ == "__main__":
#     app.run_server(debug=True)
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # Load real datasets
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EM = pd.read_csv('ethylene_methane.txt', sep='\\s+', skiprows=[0], header=None)

# # Rename columns
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]
# EM.columns = ['Time (seconds)', 'Methane conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Initialize the app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Gas Sensor Dashboard", style={'text-align': 'center'}),
    
#     # Dropdown to select gas type
#     html.Div([
#         html.Label("Select Gas Type:"),
#         dcc.Dropdown(
#             id='gas-type',
#             options=[
#                 {'label': 'CO2 and Ethylene', 'value': 'EC'},
#                 {'label': 'Methane and Ethylene', 'value': 'EM'}
#             ],
#             value='EC',
#             clearable=False
#         )
#     ], style={'margin-bottom': '20px'}),
    
#     # Graphs
#     dcc.Graph(id='gas-concentration-graph'),
#     dcc.Graph(id='sensor-response-graph'),
#     dcc.Graph(id='anomalies-graph')
# ])

# # Callbacks
# @app.callback(
#     [Output('gas-concentration-graph', 'figure'),
#      Output('sensor-response-graph', 'figure'),
#      Output('anomalies-graph', 'figure')],
#     [Input('gas-type', 'value')]
# )
# def update_dashboard(selected_gas):
#     data = EC if selected_gas == 'EC' else EM
    
#     # Gas concentration graph
#     fig1 = px.line(
#         data, x='Time (seconds)', y=data.columns[1:3],
#         title='Gas Concentrations Over Time',
#         labels={'value': 'Concentration (ppm)', 'variable': 'Gas'}
#     )
    
#     # Sensor response graph
#     fig2 = px.line(
#         data, x='Time (seconds)', y=[f'Sensor{i}' for i in range(1, 5)],
#         title='Sensor Responses Over Time',
#         labels={'value': 'Sensor Reading', 'variable': 'Sensor'}
#     )
    
#     # Anomalies detection
#     anomalies = data[[f'Sensor{i}' for i in range(1, 5)]].apply(lambda x: x > x.mean() + 2 * x.std(), axis=0)
#     anomaly_counts = anomalies.sum()
#     fig3 = px.bar(
#         anomaly_counts,
#         title='Anomalies Detected Across Sensors',
#         labels={'index': 'Sensor', 'value': 'Anomaly Count'}
#     )
    
#     return fig1, fig2, fig3

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from flask_caching import Cache


# # Load real datasets

# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EM = pd.read_csv('ethylene_methane.txt', sep='\\s+', skiprows=[0], header=None)

# # Rename columns
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]
# EM.columns = ['Time (seconds)', 'Methane conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]


# # Initialize the app
# app = dash.Dash(__name__)


# # Layout


# app.layout = html.Div([
#     html.H1("Enhanced Gas Sensor Dashboard", style={'text-align': 'center'}),
#     # Dropdown to select gas type
    
#     html.Div([
#         html.Label("Select Gas Type:"),
#         dcc.Dropdown(
#             id='gas-type',
#             options=[
#                 {'label': 'CO2 and Ethylene', 'value': 'EC'},
#                 {'label': 'Methane and Ethylene', 'value': 'EM'}
#             ],
#             value='EC',
#             clearable=False
#         )
#     ], style={'margin-bottom': '20px'}),

#     # Graphs
    
#     dcc.Graph(id='gas-concentration-graph'),
#     dcc.Graph(id='sensor-response-graph'),
#     dcc.Graph(id='anomalies-graph'),
#     dcc.Graph(id='sensor-comparison-graph'),
#     dcc.Graph(id='sensor-density-plot'),
#     dcc.Graph(id='heatmap-graph')
    
# ])

# # Callbacks
# @app.callback(
#     [Output('gas-concentration-graph', 'figure'),
#      Output('sensor-response-graph', 'figure'),
#      Output('anomalies-graph', 'figure'),
#      Output('sensor-comparison-graph', 'figure'),
#      Output('sensor-density-plot', 'figure'),
#      Output('heatmap-graph', 'figure')],
#     [Input('gas-type', 'value')]
# )

# def update_dashboard(selected_gas):
#     data = EC if selected_gas == 'EC' else EM

#     # Gas concentration graph
#     fig1 = px.line(
#         data, x='Time (seconds)', y=data.columns[1:3],
#         title='Gas Concentrations Over Time',
#         labels={'value': 'Concentration (ppm)', 'variable': 'Gas'}
#     )

#     # Sensor response graph
#     fig2 = px.line(
#         data, x='Time (seconds)', y=[f'Sensor{i}' for i in range(1, 5)],
#         title='Sensor Responses Over Time',
#         labels={'value': 'Sensor Reading', 'variable': 'Sensor'}
#     )

#     # Anomalies detection
#     anomalies = data[[f'Sensor{i}' for i in range(1, 17)]].apply(lambda x: x > x.mean() + 2 * x.std(), axis=0)
#     anomaly_counts = anomalies.sum()
#     fig3 = px.bar(
#         anomaly_counts,
#         title='Anomalies Detected Across Sensors',
#         labels={'index': 'Sensor', 'value': 'Anomaly Count'}
#     )

#     # Sensor comparison graph
#     fig4 = px.line(
#         data, x='Time (seconds)', y=[f'Sensor{i}' for i in range(1, 17)],
#         title='Comparison of All Sensors',
#         labels={'value': 'Sensor Reading', 'variable': 'Sensor'}
#     )

#     # Sensor density plot
#     fig5 = px.histogram(
#         data, x=[f'Sensor{i}' for i in range(1, 5)],
#         nbins=50, title='Density Distribution of Sensor Values',
#         labels={'value': 'Sensor Reading', 'variable': 'Sensor'}
#     )

#     # Heatmap
#     corr = data[[f'Sensor{i}' for i in range(1, 17)]].corr()
#     fig6 = px.imshow(
#         corr, text_auto=True, color_continuous_scale='Blues',
#         title='Heatmap of Sensor Correlations'
#     )

#     return fig1, fig2, fig3, fig4, fig5, fig6

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # Load datasets
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EM = pd.read_csv('ethylene_methane.txt', sep='\\s+', skiprows=[0], header=None)

# # Rename columns
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]
# EM.columns = ['Time (seconds)', 'Methane conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Set index
# EC.set_index('Time (seconds)', inplace=True)
# EM.set_index('Time (seconds)', inplace=True)

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Gas Sensor Dashboard", style={'text-align': 'center'}),

#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Graph(
#                 id='gas-concentration',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Sensor Responses', children=[
#             dcc.Graph(
#                 id='sensor-responses',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Correlation Heatmap', children=[
#             dcc.Graph(
#                 id='correlation-heatmap',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Anomalies Detection', children=[
#             dcc.Graph(
#                 id='anomalies-detection',
#                 config={'displayModeBar': True}
#             )
#         ])
#     ])
# ])

# # Callbacks
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-responses', 'figure'),
#      Output('correlation-heatmap', 'figure'),
#      Output('anomalies-detection', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(EC, y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], title='Gas Concentrations Over Time')

#     # Sensor Responses
#     fig2 = px.line(EC, y=[f'Sensor{i}' for i in range(1, 5)], title='Sensor Responses Over Time')

#     # Correlation Heatmap
#     corr = EC[[f'Sensor{i}' for i in range(1, 5)]].corr()
#     fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlations')

#     # Anomalies Detection
#     anomalies = EC[[f'Sensor{i}' for i in range(1, 5)]].apply(lambda x: x > x.mean() + 2 * x.std())
#     fig4 = px.scatter(EC.reset_index(), x='Time (seconds)', y=anomalies.sum(axis=1),
#                       title='Anomalies Detection Over Time')

#     return fig1, fig2, fig3, fig4

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from flask_caching import Cache
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from scipy.signal import find_peaks

# # Load datasets once
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Flask caching setup
# app = dash.Dash(__name__)
# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory',
#     'CACHE_THRESHOLD': 50
# })

# # Prepare data for prediction
# sensor_column = 'Sensor1'
# X = EC[['Time (seconds)']]
# y = EC[sensor_column]

# # Standardize data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_scaled, y)

# # Generate future predictions
# future_time = np.linspace(X['Time (seconds)'].max() + 1, X['Time (seconds)'].max() + 100, 100).reshape(-1, 1)
# future_time_scaled = scaler.transform(future_time)
# predictions = model.predict(future_time_scaled)

# # Layout
# app.layout = html.Div([
#     html.H1("Enhanced Gas Sensor Dashboard with AI Prediction", style={'text-align': 'center'}),

#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Loading(
#                 id="loading-1",
#                 children=[dcc.Graph(id='gas-concentration')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Sensor Correlations', children=[
#             dcc.Loading(
#                 id="loading-2",
#                 children=[dcc.Graph(id='sensor-correlation')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Distribution Analysis', children=[
#             dcc.Loading(
#                 id="loading-3",
#                 children=[dcc.Graph(id='distribution-analysis')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Peak Detection', children=[
#             dcc.Loading(
#                 id="loading-4",
#                 children=[dcc.Graph(id='peak-detection')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Anomaly Detection', children=[
#             dcc.Loading(
#                 id="loading-5",
#                 children=[dcc.Graph(id='anomaly-detection')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Trend Analysis', children=[
#             dcc.Loading(
#                 id="loading-6",
#                 children=[dcc.Graph(id='trend-analysis')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='AI Prediction', children=[
#             dcc.Loading(
#                 id="loading-7",
#                 children=[dcc.Graph(id='ai-prediction')],
#                 type="circle"
#             )
#         ])
#     ])
# ])

# # Callbacks for graphs
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-correlation', 'figure'),
#      Output('distribution-analysis', 'figure'),
#      Output('peak-detection', 'figure'),
#      Output('anomaly-detection', 'figure'),
#      Output('trend-analysis', 'figure'),
#      Output('ai-prediction', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(EC, x='Time (seconds)', y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], 
#                    title='Gas Concentrations Over Time')

#     # Sensor Correlation Heatmap
#     corr = EC[[f'Sensor{i}' for i in range(1, 17)]].corr()
#     fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlation Heatmap')

#     # Distribution Analysis
#     fig3 = px.histogram(EC, x=[f'Sensor{i}' for i in range(1, 5)], nbins=50,
#                         title='Sensor Value Distribution')

#     # Peak Detection
#     sensor_data = EC['Sensor1']
#     peaks, _ = find_peaks(sensor_data, height=sensor_data.mean() + sensor_data.std())
#     fig4 = go.Figure()
#     fig4.add_trace(go.Scatter(x=EC['Time (seconds)'], y=sensor_data, mode='lines', name='Sensor1'))
#     fig4.add_trace(go.Scatter(x=EC['Time (seconds)'][peaks], y=sensor_data.iloc[peaks], 
#                               mode='markers', name='Peaks', marker=dict(color='red')))
#     fig4.update_layout(title='Peak Detection for Sensor1')

#     # Anomaly Detection
#     anomalies = sensor_data > sensor_data.mean() + 2 * sensor_data.std()
#     fig5 = px.scatter(x=EC['Time (seconds)'], y=sensor_data, color=anomalies,
#                       title='Anomaly Detection in Sensor1', labels={'color': 'Anomaly'})

#     # Trend Analysis
#     trend = sensor_data.rolling(window=50).mean()
#     fig6 = go.Figure()
#     fig6.add_trace(go.Scatter(x=EC['Time (seconds)'], y=sensor_data, mode='lines', name='Sensor1'))
#     fig6.add_trace(go.Scatter(x=EC['Time (seconds)'], y=trend, mode='lines', name='Trend', line=dict(color='red')))
#     fig6.update_layout(title='Trend Analysis for Sensor1')

#     # AI Prediction
#     fig7 = go.Figure()
#     fig7.add_trace(go.Scatter(x=EC['Time (seconds)'], y=sensor_data, mode='lines', name='Actual Data'))
#     fig7.add_trace(go.Scatter(x=future_time.flatten(), y=predictions, mode='lines', name='Predicted Data',
#                               line=dict(color='green')))
#     fig7.update_layout(title='AI Prediction for Future Data')

#     return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from scipy.signal import find_peaks
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model

# # Generate synthetic data if not available
# try:
#     synthetic_data = pd.read_csv('synthetic_gas_chromatography_data.csv')
#     print("Synthetic data loaded successfully.")
# except FileNotFoundError:
#     print("Synthetic data not found. Generating new data...")
#     time_steps = 1000
#     time = np.arange(0, time_steps)
#     co2_conc = np.sin(0.01 * time) * 10 + np.random.normal(0, 0.5, size=time_steps)
#     ethylene_conc = np.cos(0.01 * time) * 10 + np.random.normal(0, 0.5, size=time_steps)
#     sensor_data = {f"Sensor{i}": np.sin(0.01 * time + i) * 5 + np.random.normal(0, 0.5, size=time_steps) for i in range(1, 17)}
#     synthetic_data = pd.DataFrame({
#         'Time (seconds)': time,
#         'CO2 conc (ppm)': co2_conc,
#         'Ethylene conc (ppm)': ethylene_conc,
#         **sensor_data
#     })
#     synthetic_data.to_csv('synthetic_gas_chromatography_data.csv', index=False)
#     print("Synthetic data saved as 'synthetic_gas_chromatography_data.csv'.")

# # Load pre-trained model
# model = load_model('trained_lstm_model.h5')
# print("Loaded pre-trained model.")

# # Prepare data
# sequence_length = 50
# sensor_column = 'Sensor1'
# scaler = MinMaxScaler()
# sensor_data = synthetic_data[[sensor_column]].values
# scaled_data = scaler.fit_transform(sensor_data)

# # Generate future predictions
# future_input = scaled_data[-sequence_length:]
# future_predictions = []
# for _ in range(100):
#     future_pred = model.predict(future_input.reshape(1, -1, 1))
#     future_predictions.append(future_pred[0, 0])
#     future_input = np.append(future_input[1:], future_pred, axis=0)

# future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# future_time = np.arange(synthetic_data['Time (seconds)'].iloc[-1] + 1, 
#                         synthetic_data['Time (seconds)'].iloc[-1] + 1 + len(future_predictions))

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Enhanced Gas Chromatography Dashboard", style={'text-align': 'center'}),
#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Graph(id='gas-concentration')
#         ]),
#         dcc.Tab(label='Sensor Correlations', children=[
#             dcc.Graph(id='sensor-correlation')
#         ]),
#         dcc.Tab(label='Distribution Analysis', children=[
#             dcc.Graph(id='distribution-analysis')
#         ]),
#         dcc.Tab(label='Peak Detection', children=[
#             dcc.Graph(id='peak-detection')
#         ]),
#         dcc.Tab(label='Anomaly Detection', children=[
#             dcc.Graph(id='anomaly-detection')
#         ]),
#         dcc.Tab(label='Trend Analysis', children=[
#             dcc.Graph(id='trend-analysis')
#         ]),
#         dcc.Tab(label='AI Prediction', children=[
#             dcc.Graph(id='ai-prediction')
#         ])
#     ])
# ])

# # Callbacks for updating graphs
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-correlation', 'figure'),
#      Output('distribution-analysis', 'figure'),
#      Output('peak-detection', 'figure'),
#      Output('anomaly-detection', 'figure'),
#      Output('trend-analysis', 'figure'),
#      Output('ai-prediction', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(synthetic_data, x='Time (seconds)', y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], 
#                    title='Gas Concentrations Over Time')

#     # Sensor Correlations
#     corr = synthetic_data[[f"Sensor{i}" for i in range(1, 17)]].corr()
#     fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlation Heatmap')

#     # Distribution Analysis
#     fig3 = px.histogram(synthetic_data, x=[f"Sensor{i}" for i in range(1, 5)], nbins=50,
#                         title='Sensor Value Distribution')

#     # Peak Detection
#     peaks, _ = find_peaks(sensor_data.flatten(), height=sensor_data.mean() + sensor_data.std())
#     fig4 = go.Figure()
#     fig4.add_trace(go.Scatter(x=synthetic_data['Time (seconds)'], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
#     fig4.add_trace(go.Scatter(x=synthetic_data['Time (seconds)'].iloc[peaks], 
#                               y=sensor_data[peaks], mode='markers', name='Peaks', marker=dict(color='red')))
#     fig4.update_layout(title='Peak Detection for Sensor1')

#     # Anomaly Detection
#     anomalies = sensor_data.flatten() > sensor_data.mean() + 2 * sensor_data.std()
#     fig5 = px.scatter(x=synthetic_data['Time (seconds)'], y=sensor_data.flatten(), color=anomalies,
#                       title='Anomaly Detection in Sensor1', labels={'color': 'Anomaly'})

#     # Trend Analysis
#     trend = pd.Series(sensor_data.flatten()).rolling(window=50).mean()
#     fig6 = go.Figure()
#     fig6.add_trace(go.Scatter(x=synthetic_data['Time (seconds)'], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
#     fig6.add_trace(go.Scatter(x=synthetic_data['Time (seconds)'], y=trend, mode='lines', name='Trend', line=dict(color='red')))
#     fig6.update_layout(title='Trend Analysis for Sensor1')

#     # AI Prediction
#     fig7 = go.Figure()
#     fig7.add_trace(go.Scatter(x=synthetic_data['Time (seconds)'], y=sensor_data.flatten(), mode='lines', name='Actual Data'))
#     fig7.add_trace(go.Scatter(x=future_time, y=future_predictions.flatten(), mode='lines', name='Future Predictions',
#                               line=dict(color='green', dash='dash')))
#     fig7.update_layout(title='AI Prediction for Future Data')

#     return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # Load datasets
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EM = pd.read_csv('ethylene_methane.txt', sep='\\s+', skiprows=[0], header=None)

# # Rename columns
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]
# EM.columns = ['Time (seconds)', 'Methane conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Set index
# EC.set_index('Time (seconds)', inplace=True)
# EM.set_index('Time (seconds)', inplace=True)

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Gas Sensor Dashboard", style={'text-align': 'center'}),

#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Graph(
#                 id='gas-concentration',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Sensor Responses', children=[
#             dcc.Graph(
#                 id='sensor-responses',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Correlation Heatmap', children=[
#             dcc.Graph(
#                 id='correlation-heatmap',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Anomalies Detection', children=[
#             dcc.Graph(
#                 id='anomalies-detection',
#                 config={'displayModeBar': True}
#             )
#         ])
#     ])
# ])

# # Callbacks
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-responses', 'figure'),
#      Output('correlation-heatmap', 'figure'),
#      Output('anomalies-detection', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(EC, y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], title='Gas Concentrations Over Time')

#     # Sensor Responses
#     fig2 = px.line(EC, y=[f'Sensor{i}' for i in range(1, 5)], title='Sensor Responses Over Time')

#     # Correlation Heatmap
#     corr = EC[[f'Sensor{i}' for i in range(1, 17)]].corr()
#     fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlations')

#     # Anomalies Detection
#     anomalies = EC[[f'Sensor{i}' for i in range(1, 5)]].apply(lambda x: x > x.mean() + 2 * x.std())
#     fig4 = px.scatter(EC.reset_index(), x='Time (seconds)', y=anomalies.sum(axis=1),
#                       title='Anomalies Detection Over Time')

#     return fig1, fig2, fig3, fig4

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)
    

# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import numpy as np
# from scipy.signal import find_peaks

# # Load datasets
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EM = pd.read_csv('ethylene_methane.txt', sep='\\s+', skiprows=[0], header=None)

# # Rename columns
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]
# EM.columns = ['Time (seconds)', 'Methane conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Set index
# EC.set_index('Time (seconds)', inplace=True)
# EM.set_index('Time (seconds)', inplace=True)

# # Simulate GC Data for Anomalies Detection
# np.random.seed(42)
# time = np.linspace(0, 100, 1000)
# peak_positions = [20, 50, 80]
# peak_amplitudes = [1, 1.5, 1.2]
# normal_peaks = sum(
#     amp * np.exp(-0.5 * ((time - pos) ** 2) / 1) for pos, amp in zip(peak_positions, peak_amplitudes)
# )
# shifted_peaks = sum(
#     amp * np.exp(-0.5 * ((time - (pos + np.random.uniform(1, 5))) ** 2) / 1)
#     for pos, amp in zip(peak_positions, peak_amplitudes)
# )
# noise = np.random.normal(0, 0.05, size=time.shape)
# signal = normal_peaks + shifted_peaks + noise

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Gas Sensor Dashboard", style={'text-align': 'center'}),

#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Graph(
#                 id='gas-concentration',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Sensor Responses', children=[
#             dcc.Graph(
#                 id='sensor-responses',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Correlation Heatmap', children=[
#             dcc.Graph(
#                 id='correlation-heatmap',
#                 config={'displayModeBar': True}
#             )
#         ]),
#         dcc.Tab(label='Anomalies Detection', children=[
#             html.Div([
#                 dcc.Graph(id='anomalies-detection'),
#                 html.Label("Noise Threshold:"),
#                 dcc.Slider(
#                     id="noise-threshold",
#                     min=0,
#                     max=0.2,
#                     step=0.01,
#                     value=0.1,
#                     marks={i: str(round(i, 2)) for i in np.arange(0, 0.21, 0.05)}
#                 )
#             ], style={"padding": "20px"})
#         ])
#     ])
# ])

# # Callbacks for graphs
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-responses', 'figure'),
#      Output('correlation-heatmap', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(EC, y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], title='Gas Concentrations Over Time')

#     # Sensor Responses
#     fig2 = px.line(EC, y=[f'Sensor{i}' for i in range(1, 5)], title='Sensor Responses Over Time')

#     # Correlation Heatmap
#     corr = EC[[f"Sensor{i}" for i in range(1, 17)]].corr()
#     fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlations')

#     return fig1, fig2, fig3


# @app.callback(
#     Output("anomalies-detection", "figure"),
#     [Input("noise-threshold", "value")]
# )
# def update_anomalies_graph(threshold):
#     filtered_peaks, _ = find_peaks(signal, height=threshold, distance=50)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time, y=signal, mode="lines", name="Signal", line=dict(color="blue")))
#     fig.add_trace(go.Scatter(
#         x=time[filtered_peaks], y=signal[filtered_peaks],
#         mode="markers", name="Detected Peaks", marker=dict(color="red", size=10)
#     ))

#     fig.update_layout(
#         title="Simulated GC Data Analysis",
#         xaxis_title="Time",
#         yaxis_title="Intensity",
#         template="plotly_white"
#     )
#     return fig

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)


# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from scipy.signal import find_peaks
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # Load the pre-trained model
# model = load_model('trained_lstm_model.h5')
# print("Loaded pre-trained model.")

# # Load dataset
# EC = pd.read_csv('ethylene_CO.txt', sep='\\s+', skiprows=[0], header=None)
# EC.columns = ['Time (seconds)', 'CO2 conc (ppm)', 'Ethylene conc (ppm)'] + [f'Sensor{i}' for i in range(1, 17)]

# # Prepare data for predictions
# sensor_column = 'Sensor1'
# time_column = 'Time (seconds)'
# scaler = MinMaxScaler()
# sensor_data = EC[[sensor_column]].values
# scaled_data = scaler.fit_transform(sensor_data)

# # Generate future predictions
# sequence_length = 50
# future_time = np.linspace(EC[time_column].max() + 1, EC[time_column].max() + 100, 100).reshape(-1, 1)
# future_input = scaled_data[-sequence_length:]
# predictions = []

# for _ in range(100):
#     future_pred = model.predict(future_input.reshape(1, -1, 1))
#     predictions.append(future_pred[0, 0])
#     future_input = np.append(future_input[1:], future_pred, axis=0)

# predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Layout
# app.layout = html.Div([
#     html.H1("Enhanced Gas Sensor Dashboard with LSTM Predictions", style={'text-align': 'center'}),
#     dcc.Tabs([
#         dcc.Tab(label='Gas Concentrations', children=[
#             dcc.Loading(
#                 id="loading-1",
#                 children=[dcc.Graph(id='gas-concentration')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Sensor Correlations', children=[
#             dcc.Loading(
#                 id="loading-2",
#                 children=[dcc.Graph(id='sensor-correlation')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Distribution Analysis', children=[
#             dcc.Loading(
#                 id="loading-3",
#                 children=[dcc.Graph(id='distribution-analysis')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Peak Detection', children=[
#             dcc.Loading(
#                 id="loading-4",
#                 children=[dcc.Graph(id='peak-detection')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Anomaly Detection', children=[
#             dcc.Loading(
#                 id="loading-5",
#                 children=[dcc.Graph(id='anomaly-detection')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='Trend Analysis', children=[
#             dcc.Loading(
#                 id="loading-6",
#                 children=[dcc.Graph(id='trend-analysis')],
#                 type="circle"
#             )
#         ]),
#         dcc.Tab(label='AI Prediction', children=[
#             dcc.Loading(
#                 id="loading-7",
#                 children=[dcc.Graph(id='ai-prediction')],
#                 type="circle"
#             )
#         ])
#     ])
# ])

# # Callbacks for graphs
# @app.callback(
#     [Output('gas-concentration', 'figure'),
#      Output('sensor-correlation', 'figure'),
#      Output('distribution-analysis', 'figure'),
#      Output('peak-detection', 'figure'),
#      Output('anomaly-detection', 'figure'),
#      Output('trend-analysis', 'figure'),
#      Output('ai-prediction', 'figure')],
#     [Input('gas-concentration', 'id')]
# )
# def update_graphs(tab_selected):
#     # Gas Concentrations
#     fig1 = px.line(EC, x=time_column, y=['CO2 conc (ppm)', 'Ethylene conc (ppm)'], 
#                    title='Gas Concentrations Over Time')

#     # Sensor Correlation Heatmap
#     corr = EC[[f'Sensor{i}' for i in range(1, 17)]].corr()
#     fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues', title='Sensor Correlation Heatmap')

#     # Distribution Analysis
#     fig3 = px.histogram(EC, x=[f'Sensor{i}' for i in range(1, 5)], nbins=50,
#                         title='Sensor Value Distribution')

#     # Peak Detection
#     peaks, _ = find_peaks(sensor_data.flatten(), height=sensor_data.mean() + sensor_data.std())
#     fig4 = go.Figure()
#     fig4.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
#     fig4.add_trace(go.Scatter(x=EC[time_column].iloc[peaks], y=sensor_data[peaks], 
#                               mode='markers', name='Peaks', marker=dict(color='red')))
#     fig4.update_layout(title='Peak Detection for Sensor1')

#     # Anomaly Detection
#     anomalies = sensor_data.flatten() > sensor_data.mean() + 2 * sensor_data.std()
#     fig5 = px.scatter(x=EC[time_column], y=sensor_data.flatten(), color=anomalies,
#                       title='Anomaly Detection in Sensor1', labels={'color': 'Anomaly'})

#     # Trend Analysis
#     trend = pd.Series(sensor_data.flatten()).rolling(window=50).mean()
#     fig6 = go.Figure()
#     fig6.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Sensor1'))
#     fig6.add_trace(go.Scatter(x=EC[time_column], y=trend, mode='lines', name='Trend', line=dict(color='red')))
#     fig6.update_layout(title='Trend Analysis for Sensor1')

#     # AI Prediction
#     fig7 = go.Figure()
#     fig7.add_trace(go.Scatter(x=EC[time_column], y=sensor_data.flatten(), mode='lines', name='Actual Data'))
#     fig7.add_trace(go.Scatter(x=future_time.flatten(), y=predictions.flatten(), mode='lines', name='Predicted Data',
#                               line=dict(color='green')))
#     fig7.update_layout(title='AI Prediction for Future Data')

#     return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)

import matplotlib.pyplot as plt

# Define the stages of the flow
stages = [
    "User Input: Load GC Data",
    "Data Preprocessing: Clean & Normalize",
    "AI Analysis: Peak Detection & Prediction",
    "Visualization: Dashboards (Gas, Anomalies, Correlations)",
    "Insights Delivery: Actionable Reports"
]

# Assign positions for the flow stages
positions = range(len(stages))

# Create the flow chart
plt.figure(figsize=(10, 6))
plt.barh(positions, [1] * len(stages), color="lightblue", edgecolor="black")
plt.yticks(positions, stages, fontsize=12)
plt.xlabel("Flow Process", fontsize=14)
plt.title("User Journey for Fullmetal Alchemist Project", fontsize=16)

# Add arrows between the stages
for i in range(len(stages) - 1):
    plt.arrow(0.5, i, 0, -0.7, head_width=0.1, head_length=0.2, fc='black', ec='black', length_includes_head=True)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
