import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)
fig = px.line(x=[1, 2, 3], y=[3, 1, 6])

app.layout = html.Div([
    html.H1("My Dash App"),
    dcc.Graph(figure=fig)
])

server = app.server  # Expose the server variable for Render

if __name__ == '__main__':
    app.run(debug=True)
