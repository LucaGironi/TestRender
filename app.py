import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_extensions.enrich as de
from plotly.subplots import make_subplots

# Constants
r_in = 1.0  # mm
r_out = 10.0  # mm
mu_e = 36000e-4  # mm^2/V/s
mu_h = 42000e-4  # mm^2/V/s
q = 1.6e-19  # C
dt = 0.001  # s
total_time = 0.1  # s (updated)
steps = int(total_time / dt)

def E_r(r, V):
    return V / (r * np.log(r_out / r_in))

def weighting_field(r):
    return 1 / r

def generate_initial_positions(n):
    r = np.random.uniform(r_in + 0.5, r_out - 0.5, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    return r, theta

def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)

app = de.Dash(__name__)
server = app.server

default_n = 15
r0, theta0 = generate_initial_positions(default_n)

app.layout = html.Div([
    html.H3("Coaxial HPGe Detector: Drift and Induced Current"),
    dcc.Graph(id='drift-plot'),
    html.Div([
        html.Button("Start Simulation", id='start-button', n_clicks=0),
        html.Button("Reset", id='reset-button', n_clicks=0, style={'marginLeft': '10px'}),
    ]),
    html.Label("Bias Voltage (V):"),
    dcc.Slider(id='bias-slider', min=500, max=5000, step=100, value=3000,
               marks={i: str(i) for i in range(500, 5500, 1000)}),
    html.Label("Number of Particle Pairs:"),
    dcc.Slider(id='count-slider', min=5, max=50, step=1, value=default_n,
               marks={i: str(i) for i in range(5, 55, 10)}),
    dcc.Interval(id='interval', interval=200, n_intervals=0, disabled=True),
    dcc.Store(id='frame-store', data=0),
    dcc.Store(id='r-store', data=r0.tolist()),
    dcc.Store(id='theta-store', data=theta0.tolist()),
    dcc.Store(id='current-data', data={'time': [], 'Ie': [], 'Ih': [], 'Itotal': []}),
])

@app.callback(
    Output('r-store', 'data'),
    Output('theta-store', 'data'),
    Output('frame-store', 'data'),
    Output('interval', 'disabled'),
    Output('current-data', 'data'),
    Input('reset-button', 'n_clicks'),
    State('count-slider', 'value'),
)
def reset_simulation(_, num_pairs):
    r, theta = generate_initial_positions(num_pairs)
    return r.tolist(), theta.tolist(), 0, True, {'time': [], 'Ie': [], 'Ih': [], 'Itotal': []}

@app.callback(
    Output('interval', 'disabled'),
    Output('frame-store', 'data'),
    Input('start-button', 'n_clicks'),
    State('frame-store', 'data'),
)
def start_simulation(_, frame):
    return False, 0

@app.callback(
    Output('drift-plot', 'figure'),
    Output('frame-store', 'data'),
    Output('current-data', 'data'),
    Input('interval', 'n_intervals'),
    State('frame-store', 'data'),
    State('r-store', 'data'),
    State('theta-store', 'data'),
    State('bias-slider', 'value'),
    State('current-data', 'data'),
)
def update_plot(_, frame, r_init, theta, V, current_data):
    if frame >= steps:
        return dash.no_update, frame, current_data

    r_init = np.array(r_init)
    theta = np.array(theta)
    time = frame * dt
    Er_init = E_r(r_init, V)

    r_e = np.maximum(r_init - mu_e * Er_init * time, r_in)
    r_h = np.minimum(r_init + mu_h * Er_init * time, r_out)

    xe, ye = polar_to_cartesian(r_e, theta)
    xh, yh = polar_to_cartesian(r_h, theta)

    # Compute induced current
    ve = mu_e * E_r(r_e, V)
    vh = mu_h * E_r(r_h, V)
    we = weighting_field(r_e)
    wh = weighting_field(r_h)

    Ie = np.sum(q * ve * we)
    Ih = np.sum(q * vh * wh)
    Itotal = Ie + Ih

    current_data['time'].append(time)
    current_data['Ie'].append(Ie * 1e12)
    current_data['Ih'].append(Ih * 1e12)
    current_data['Itotal'].append(Itotal * 1e12)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Charge Carrier Drift", "Induced Current"),
        column_widths=[0.5, 0.5]
    )

    # Left plot: drift
    fig.add_trace(go.Scatter(x=xe, y=ye, mode='markers', marker=dict(color='blue', size=8), name='Electrons'), row=1, col=1)
    fig.add_trace(go.Scatter(x=xh, y=yh, mode='markers', marker=dict(color='red', size=8), name='Holes'), row=1, col=1)

    fig.update_xaxes(title_text="x (mm)", range=[-r_out - 1, r_out + 1], row=1, col=1)
    fig.update_yaxes(title_text="y (mm)", range=[-r_out - 1, r_out + 1], scaleanchor="x", scaleratio=1, row=1, col=1)

    # Right plot: current
    fig.add_trace(go.Scatter(x=current_data['time'], y=current_data['Ie'],
                             mode='lines', name='Electrons (pA)', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=current_data['time'], y=current_data['Ih'],
                             mode='lines', name='Holes (pA)', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=current_data['time'], y=current_data['Itotal'],
                             mode='lines', name='Total (pA)', line=dict(color='green')), row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Current (pA)", row=1, col=2)

    fig.update_layout(
        height=600,
        title=f"Time: {time:.3f} s — Bias: {V} V",
        showlegend=True
    )

    return fig, frame + 1, current_data

if __name__ == '__main__':
    app.run(debug=True)
