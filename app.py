import pandas as pd
import numpy as np
from helper_functions import adjust_stat
from copy import copy
import pickle
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

from data_processing import get_batter_df, get_pitcher_df, get_pitch_characteristics_df, load_models
from sample_pitches import sample_pas
from dashboard_functions import get_summary_stats, get_pitches_summary

import dash
from dash import dcc, html, Output, Input, State, dash_table, ctx, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

batter_df = get_batter_df()
pitcher_df = get_pitcher_df()
pitch_characteristics_df = get_pitch_characteristics_df()

models = load_models()
data = {'batter_df':batter_df, 
        'pitcher_df':pitcher_df,
        'pitch_characteristics_df':pitch_characteristics_df}

pitcher_index = pd.read_parquet('data/pitcher_index.parquet').sort_values(by='name')
pitcher_options = [{'label': name, 'value': id} for id, name in zip(pitcher_index['id'], pitcher_index['name'])]
pitcher_dict = {id: name for id, name in zip(pitcher_index['id'], pitcher_index['name'])}

batter_index = pd.read_parquet('data/batter_index.parquet').sort_values(by='name')
batter_options = [{'label': name, 'value': id} for id, name in zip(batter_index['id'], batter_index['name'])]
batter_dict = {id: name for id, name in zip(batter_index['id'], batter_index['name'])}

pitch_colors = {
    'FF':'#c13d4d',
    'FA':'#c13d4d',
    'SI':'#f0a139', 
    'FC':'#894432', 
    'CH':'#5abb4e',
    'FS':'#5daaab', 
    'FO':'#76c9ad',
    'SC':'#82d852',
    'CU':'#5fcee9',
    'EP':'#5fcee9',
    'KC':'#5c38c5',
    'CS':'#2b66f6',
    'SL':'#ede750',
    'ST':'#d6b552',
    'SV':'#98aed1',
    'KN':'#3e44c5'
}

pitch_names = {
    'FF':'Fastball (4-seam)',
    'FA':'Fastball',
    'SI':'Sinker (2-seam)', 
    'FC':'Cutter', 
    'CH':'Changeup',
    'FS':'Split-finger', 
    'FO':'Forkball',
    'SC':'Screwball',
    'CU':'Curveball',
    'EP':'Eephus',
    'KC':'Knuckle Curve',
    'CS':'Slow Curve',
    'SL':'Slider',
    'ST':'Sweeper',
    'SV':'Slurve',
    'KN':'Knuckleball'
}

# Initialize the Dash app
app = dash.Dash(__name__)

server = app.server


app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Store(id='applied-filters'),

    html.Div(id='input-screen',
             children=[
                html.H1("MLB Matchup Simulator", style={'margin':'0', 'margin-top':'30px', 'margin-left':'20px'}),
                html.P("Created by Timothy Clay", style={'margin':'0', 'margin-top':'10px', 'margin-left':'20px'}),
                html.Div(
                    id='input-screen-container',
                    style={
                        'margin-top':'20px',
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center', 
                        'textAlign': 'center'
                    },
                    children=[

                        # Left: Pitcher headshot
                        html.Div(
                            id="input-pitcher-headshot",
                            style={'width': '150px'}
                        ),

                        # Middle: your original input-screen box
                        html.Div(
                            style={
                                'width': '350px',
                                'textAlign': 'center'
                            },
                            children=[
                                html.H2(
                                    "Choose Matchup",
                                    style={'marginBottom': '20px', 'color': '#333'}
                                ),

                                html.Label("Pitcher", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='pitcher-dropdown',
                                    options=pitcher_options,
                                    placeholder='Select a pitcher',
                                    style={'marginBottom': '15px'}
                                ),

                                html.Label("Batter", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='batter-dropdown',
                                    options=batter_options,
                                    placeholder='Select a batter',
                                    style={'marginBottom': '20px'}
                                ),

                                dcc.Loading(
                                    children=[
                                        html.Div(
                                            id='loading-overlay-output',
                                            children=[
                                            html.Button(
                                                'Run Model',
                                                id='run-model',
                                                n_clicks=0,
                                                style={
                                                    'backgroundColor': '#349eeb',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'padding': '10px 20px',
                                                    'textAlign': 'center',
                                                    'fontSize': '16px',
                                                    'borderRadius': '5px'
                                                }
                                            )
                                        ])
                                        ],
                                        color='gray',
                                        type="dot",
                                ),
                            ]
                        ),

                        # Right: Batter headshot
                        html.Div(
                            id="input-batter-headshot",
                            style={'width': '150px'}
                        )
                    ]),
            ]),
    
    html.Div(id='output-screen', style={'display': 'none', 'flexDirection':'row', 'width':'100%', 'height':'100%'}, children=[

        html.Div(style={'width':'50%', 'justifyContent': 'center', 'textAlign': 'center', 'paddingTop':'20px'}, 
                 children=[
            html.Div(
                style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'width': '100%',
                    'gap': '20px',
                    "marginBottom": "10px",  
                    "paddingBottom": "10px"
                },
                children=[
                    html.Div(id='pitcher-headshot-container', style={'flex': '0 0 auto'}),
                    html.Div(
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'textAlign': 'center'
                        },
                        children=[
                            html.Div(id="pitcher-name-container"),
                            html.P("vs."),
                            html.Div(id="batter-name-container")
                        ]
                    ),
                    html.Div(id='batter-headshot-container', style={'flex': '0 0 auto'})
                ]
            ),
            html.Div(id='summary-output',
                     style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'width': '100%'}),
        ]), 
        html.Div(style={'width':'50%', 'justifyContent': 'center', 'alignItems': 'center', 'textAlign': 'center'}, children=[
            html.Div(id="summary-table", 
                     style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'width': '100%', 'margin-top':'50px'}),
            html.Div(style={'display':'flex', 'flexDirection':'row', 'justifyContent': 'center', 'alignItems': 'center'}, 
                     children=[
                         html.Div([
                             html.H3('Pitch Break'),
                             dcc.Graph(id='pitch-break-plot', 
                        style={'height':'345px', 'border': '1px solid black', 'padding-left':'5px', 'padding-bottom':'5px', 'margin-top':'5px', 'margin-right':'5px'},
                        config={
                            'modeBarButtonsToRemove': [
                                'zoom2d', 'pan2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 
                                'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
                                ],
                                'scrollZoom': False,
                                'displayModeBar': False,  
                            }), 
                         ]),
                         html.Div([
                             html.H3('Pitch Location'),
                             dcc.Graph(id='scatter-plot', 
                        style={'height':'350px', 'border': '1px solid black', 'margin-top':'5px', 'margin-left':'5px'},
                        config={
                            'modeBarButtonsToRemove': [
                                'zoom2d', 'pan2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 
                                'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian'
                                ],
                                'scrollZoom': False,
                                'displayModeBar': False,  
                            }
                ),
                         ]),
                
                
            ])
            
        ]),
        html.Button("⚙ Filters", id="open-popup", style={
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "zIndex": 1000,
                "padding": "10px 18px",
                "backgroundColor": "rgb(230, 230, 230)",   # dark slate gray
                "color": "black",
                "border": "none",
                "borderRadius": "8px",
                "fontSize": "12px",
                "fontWeight": "500",
                "cursor": "pointer",
            },),
        html.Div(
            id="popup",
            children=[
                html.Div([
                    html.H2("Filters", style={"marginBottom": "20px", "borderBottom": "2px solid #ccc", "paddingBottom": "8px"}),

                    # Pitch type filter
                    html.Div([
                        html.H4("Pitch Types", style={"marginBottom": "10px"}),
                        dcc.Checklist(
                            id='pitch-type-checklist',
                            style={"display": "grid", "gridTemplateColumns": "repeat(5, auto)", "gap": "5px"}
                        ),
                    ], style={"marginBottom": "20px"}),

                    # Presets section
                    html.Div([
                        html.H4("Count", style={"marginBottom": "10px"}),   
                        html.Div([
                            html.Button("All", id='all-counts-button'),
                            html.Button("None", id='no-counts-button'),
                            html.Button("Pitcher Ahead", id='pahead-counts-button'),
                            html.Button("Batter Ahead", id='bahead-counts-button'),
                            html.Button("Even", id='even-counts-button'),
                            html.Button("2 Strike", id='twok-counts-button'),
                            html.Button("3 Ball", id='threeb-counts-button')
                        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})
                    ], style={"marginBottom": "20px"}),

                    # Count checklist
                    html.Div([
                        dcc.Checklist(
                            id='count-checklist',
                            options=["0-0", "1-0", "2-0", "3-0", "0-1", "1-1",
                                    "2-1", "3-1", "0-2", "1-2", "2-2", "3-2"],
                            value=["0-0", "1-0", "2-0", "3-0", "0-1", "1-1",
                                    "2-1", "3-1", "0-2", "1-2", "2-2", "3-2"],
                            style={"display": "grid", "gridTemplateColumns": "repeat(4, auto)", "gap": "5px"}
                        )
                    ]),

                    # Close button
                    html.Div([
                        html.Button("Apply Filters", id="close-popup", n_clicks=0, style={
                            "background": "#4CAF50", "color": "white",
                            "border": "none", "padding": "8px 16px",
                            "borderRadius": "6px", "cursor": "pointer",
                            "fontWeight": "bold"
                        })
                    ], style={"textAlign": "right", "marginTop": "20px"})

                ], className="modal-box", style={
                    "background": "white",
                    "padding": "25px",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 20px rgba(0,0,0,0.2)",
                    "width": "70%",
                    "maxWidth": "800px",
                    "maxHeight": "80vh",
                    "overflowY": "auto"
                })
            ],
            style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100vw",
                "height": "100vh",
                "backgroundColor": "rgba(0, 0, 0, 0.5)",
                "display": "none",  # toggled via callback
                "alignItems": "center",
                "justifyContent": "center",
                "zIndex": 1000
            }
        ),
        html.Button(
            "← Back",
            id="back-button",
            style={
                "position": "fixed",
                "top": "20px",
                "left": "20px",
                "zIndex": 80,
                "padding": "10px 18px",
                "backgroundColor": "rgb(230, 230, 230)",   # dark slate gray
                "color": "black",
                "border": "none",
                "borderRadius": "8px",
                "fontSize": "12px",
                "fontWeight": "500",
                "cursor": "pointer",
            },
        ),

    ]),
    
], 
style={'fontFamily': 'Arial, sans-serif'})

# Callback to generate data and switch views
@app.callback(
    Output('data-store', 'data'),
    Output('loading-overlay-output', 'children'),
    Output('input-screen', 'style'),
    Output('output-screen', 'style'),
    Input('run-model', 'n_clicks'),
    Input('back-button', 'n_clicks'),
    State('batter-dropdown', 'value'),
    State('pitcher-dropdown', 'value'),
    prevent_initial_call=True
)
def run_model(n_clicks, back_clicks, batter, pitcher):
    context = callback_context
    if not context.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = context.triggered[0]['prop_id'].split('.')[0]

    input_style = {'display': 'block'}
    output_style = {'display': 'none'}

    if button_id == 'back-button':
        return None, dash.no_update, input_style, output_style

    if batter is None or pitcher is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    df = sample_pas(batter, pitcher, models=models, data=data)

    return df.to_dict('records'), dash.no_update, {'display': 'none'}, {'display': 'flex'}

@app.callback(
    Output('summary-output', 'children'),
    Input('scatter-plot', 'selectedData'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_summary(selectedData, stored_data, applied_filters):

    if stored_data is None:
        return html.Div("No data to summarize.")

    df = pd.DataFrame(stored_data)

    if applied_filters['pitch_types']:
        df = df[df['pitch_type'].isin(applied_filters['pitch_types'])]

    if applied_filters['counts']:
        df = df[df['count'].isin(applied_filters['counts'])]

    if selectedData is None:
        filtered_df = df
    else:
        points = selectedData['points']
        indices = [p['pointIndex'] for p in points]
        filtered_df = df.iloc[indices]

    summary_stats = get_summary_stats(filtered_df)

    summary_stats['avg'] = f"{summary_stats['avg']:.3f}".lstrip('0')
    summary_stats['obp'] = f"{summary_stats['obp']:.3f}".lstrip('0')
    summary_stats['slg'] = f"{summary_stats['slg']:.3f}".lstrip('0')
    summary_stats['ops'] = f"{summary_stats['ops']:.3f}".lstrip('0')
    summary_stats['woba'] = f"{summary_stats['woba']:.3f}".lstrip('0')

    summary_stats['1B_per_600'] = f"{summary_stats['1B_per_600']:.0f}"
    summary_stats['2B_per_600'] = f"{summary_stats['2B_per_600']:.0f}"
    summary_stats['3B_per_600'] = f"{summary_stats['3B_per_600']:.0f}"
    summary_stats['HR_per_600'] = f"{summary_stats['HR_per_600']:.0f}"
    summary_stats['BB_per_600'] = f"{summary_stats['BB_per_600']:.0f}"
    summary_stats['K_per_600'] = f"{summary_stats['K_per_600']:.0f}"

    summary_stats['k_pct'] = f"{summary_stats['k_pct']*100:.1f}"
    summary_stats['bb_pct'] = f"{summary_stats['bb_pct']*100:.1f}"
    summary_stats['swing_pct'] = f"{summary_stats['swing_pct']*100:.1f}"
    summary_stats['zswing_pct'] = f"{summary_stats['zswing_pct']*100:.1f}"
    summary_stats['oswing_pct'] = f"{summary_stats['oswing_pct']*100:.1f}"
    summary_stats['contact_pct'] = f"{summary_stats['contact_pct']*100:.1f}"
    summary_stats['zcontact_pct'] = f"{summary_stats['zcontact_pct']*100:.1f}"
    summary_stats['ocontact_pct'] = f"{summary_stats['ocontact_pct']*100:.1f}"
    summary_stats['swstr_pct'] = f"{summary_stats['swstr_pct']*100:.1f}"
    summary_stats['zone_pct'] = f"{summary_stats['zone_pct']*100:.1f}"
    

    return html.Div([
        html.H3("Summary Data"),
        dash_table.DataTable(
            data=pd.DataFrame([summary_stats])[['total_pitches', 'batted_balls', 'swings']].to_dict('records'),
            columns=[
                {"name":"Total Pitches", "id":"total_pitches"},
                {"name":"Swings", "id":"swings"},
                {"name":"Batted Balls", "id":"batted_balls"}
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid black'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '80px', 'maxWidth': '150px'}
        ),
        html.H3("Rate Stats"),
        dash_table.DataTable(
            data=pd.DataFrame([summary_stats])[['avg', 'obp', 'slg', 'ops', 'woba', 'k_pct', 'bb_pct']].to_dict('records'),
            columns=[
                {"name":"AVG", "id":"avg"},
                {"name":"OBP", "id":"obp"},
                {"name":"SLG", "id":"slg"},
                {"name":"OPS", "id":"ops"},
                {"name":"wOBA", "id":"woba"},
                {"name":"K%", "id":"k_pct"},
                {"name":"BB%", "id":"bb_pct"}
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid black'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '80px', 'maxWidth': '150px'}
        ),
        html.H3("Stats per 600 PA"),
        dash_table.DataTable(
            data=pd.DataFrame([summary_stats])[['1B_per_600', '2B_per_600', '3B_per_600', 'HR_per_600', 'BB_per_600', 'K_per_600']].to_dict('records'),
            columns=[
                {"name":"1B", "id":"1B_per_600"},
                {"name":"2B", "id":"2B_per_600"},
                {"name":"3B", "id":"3B_per_600"},
                {"name":"HR", "id":"HR_per_600"},
                {"name":"BB", "id":"BB_per_600"},
                {"name":"K", "id":"K_per_600"}
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid black'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '80px', 'maxWidth': '150px'}
        ),
        html.H3("Plate Discipline"),
        dash_table.DataTable(
            data=pd.DataFrame([summary_stats])[['swing_pct', 'zswing_pct', 'oswing_pct', 'contact_pct', 'zcontact_pct', 'ocontact_pct', 'swstr_pct', 'zone_pct']].to_dict('records'),
            columns=[
                {"name":"Swing%", "id":"swing_pct"},
                {"name":"Z-Swing%", "id":"zswing_pct"},
                {"name":"O-Swing%", "id":"oswing_pct"},
                {"name":"Contact%", "id":"contact_pct"},
                {"name":"Z-Contact%", "id":"zcontact_pct"},
                {"name":"O-Contact%", "id":"ocontact_pct"},
                {"name":"SwStr%", "id":"swstr_pct"},
                {"name":"Zone%", "id":"zone_pct"}
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid black'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '80px', 'maxWidth': '150px'}
        )
    ],
    style={'width': 'fit-content'})

@app.callback(
    Output('summary-table', 'children'),
    Input('scatter-plot', 'selectedData'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_pitches_summary_table(selectedData, stored_data, applied_filters):
    if stored_data is None:
        return html.Div("No data to summarize.")

    df = pd.DataFrame(stored_data)

    if applied_filters['pitch_types']:
        df = df[df['pitch_type'].isin(applied_filters['pitch_types'])]

    if applied_filters['counts']:
        df = df[df['count'].isin(applied_filters['counts'])]

    if selectedData is None:
        filtered_df = df
    else:
        points = selectedData['points']
        indices = [p['pointIndex'] for p in points]
        filtered_df = df.iloc[indices]


    summary_stats = get_pitches_summary(filtered_df) 

    summary_stats['pct'] = summary_stats['pct'] * 100
    summary_stats['pfx_x'] = summary_stats['pfx_x'] * 12
    summary_stats['pfx_z'] = summary_stats['pfx_z'] * 12

    summary_stats['pct'] = summary_stats['pct'].map(lambda x: f"{x:.1f}")
    summary_stats['release_speed'] = summary_stats['release_speed'].map(lambda x: f"{x:.1f}")
    summary_stats['release_spin_rate'] = summary_stats['release_spin_rate'].map(lambda x: f"{x:.0f}")
    summary_stats['pfx_x'] = summary_stats['pfx_x'].map(lambda x: f"{x:.1f}")
    summary_stats['pfx_z'] = summary_stats['pfx_z'].map(lambda x: f"{x:.1f}")

    return html.Div([
        html.H3("Pitch Data"),
        dash_table.DataTable(
            data=summary_stats.to_dict('records'),
            columns=[{"name":"Pitch Type", "id":"pitch_type"},
                    {"name":"%", "id":"pct"}, 
                    {"name":"Velocity", "id":"release_speed"}, 
                    {"name":"Spin Rate", "id":"release_spin_rate"},
                    {"name":"Horz. Break (in)", "id":"pfx_x"},
                    {"name":"Vert. Break (in)", "id":"pfx_z"},
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid black'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'textAlign': 'center'},
            style_cell={'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': '14px', 'minWidth': '50px', 'width': '80px', 'maxWidth': '150px'}
        )],
        style={'width': '89%'}
    )

@app.callback(
    Output('pitch-type-checklist', 'options'),
    Output('pitch-type-checklist', 'value'), 
    Input('data-store', 'data')
)
def update_pitch_type_checklist(stored_data):

    if stored_data is None:
        return [], []

    df = pd.DataFrame(stored_data)
    pitch_types = df.groupby('pitch_type').agg({'pitcher':'count'}).reset_index().sort_values(by=['pitcher'], ascending=False)['pitch_type'].unique()
    
    options = [{'label': pitch_names[pt], 'value': pt} for pt in pitch_types]
    return options, pitch_types

@app.callback(
    Output('pitcher-headshot-container', 'children'),
    Output('pitcher-name-container', 'children'),
    Output('batter-headshot-container', 'children'),
    Output('batter-name-container', 'children'),
    Input('pitcher-dropdown', 'value'),
    Input('batter-dropdown', 'value')
)
def update_headshots(pitcher_id, batter_id):
    pitcher_img = html.Img(
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pitcher_id}/headshot/67/current.png',
        style={'height': '150px', 'border': '1px solid black', 'borderRadius': '4px'}
    ) if pitcher_id else ""

    pitcher_name = html.H2(
        pitcher_dict[pitcher_id],
        style={'margin': '0'}
    ) if pitcher_id else ""
    
    batter_img = html.Img(
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{batter_id}/headshot/67/current.png',
        style={'height': '150px', 'border': '1px solid black', 'borderRadius': '4px'}
    ) if batter_id else ""

    batter_name = html.H2(
        batter_dict[batter_id],
        style={'margin': '0'}
    ) if batter_id else ""
    
    return pitcher_img, pitcher_name, batter_img, batter_name

@app.callback(
    Output('input-pitcher-headshot', 'children'),
    Input('pitcher-dropdown', 'value')
)
def update_input_pitcher_headshots(pitcher_id):
    pitcher_img = html.Img(
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pitcher_id}/headshot/67/current.png',
        style={'height': '150px', 'border': '1px solid black', 'borderRadius': '4px'}
    ) if pitcher_id else ""
    
    return pitcher_img

@app.callback(
    Output('input-batter-headshot', 'children'),
    Input('batter-dropdown', 'value')
)
def update_input_batter_headshots(batter_id):
    batter_img = html.Img(
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{batter_id}/headshot/67/current.png',
        style={'height': '150px', 'border': '1px solid black', 'borderRadius': '4px'}
    ) if batter_id else ""
    
    return batter_img

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_plot(stored_data, applied_filters):

    if stored_data is None:
        return px.scatter(title="No Data")
    
    df = pd.DataFrame(stored_data)

    if applied_filters['pitch_types']:
        df = df[df['pitch_type'].isin(applied_filters['pitch_types'])]

    if applied_filters['counts']:
        df = df[df['count'].isin(applied_filters['counts'])]

    x = df['plate_x'].values
    y = df['plate_z'].values

    kde = gaussian_kde(np.vstack([x, y]))
    x_grid = np.linspace(-2, 2, 100)
    y_grid = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=x_grid, y=y_grid, z=Z,
        colorscale='reds',
        showscale=False,
        opacity=0.7,
        zsmooth='best'
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=4, opacity=0, color='black'),
        name='Points'
    ))

    fig.add_shape(
        type='rect',
        x0=-0.83, x1=0.83,
        y0=df['sz_bot'].mean(), y1=df['sz_top'].mean(),
        line=dict(color='Black', width=2),
        layer='above'  
    )

    plate_x = [-0.7083, 0.7083,  0.675,   0.0, -0.675, -0.7083]
    plate_y = [    0.5,    0.5, 0.5625, 0.625, 0.5625,     0.5]

    fig.add_trace(go.Scatter(
        x=plate_x,
        y=plate_y,
        mode='lines',
        line=dict(color='black', width=2),
        fill='toself',
    ))

    fig.update_layout(
        width=280,
        height=350,
        xaxis=dict(range=[-2, 2], showticklabels=False),
        yaxis=dict(range=[0, 5], showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        dragmode='select',
        showlegend=False
    )

    fig.update_traces(hoverinfo='skip')
    
    return fig

@app.callback(
    Output('pitch-break-plot', 'figure'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_pitch_break_plot(stored_data, applied_filters):

    if stored_data is None:
        return px.scatter(title="No Data")
    
    df = pd.DataFrame(stored_data)

    if applied_filters['pitch_types']:
        df = df[df['pitch_type'].isin(applied_filters['pitch_types'])]

    if applied_filters['counts']:
        df = df[df['count'].isin(applied_filters['counts'])]

    x = df['pfx_x'].values
    y = df['pfx_z'].values

    fig = go.Figure()

    for pitch in df['pitch_type'].unique():
        sub_df = df.loc[df['pitch_type'] == pitch]
        fig.add_trace(go.Scatter(
            x=sub_df['pfx_x']*12, y=sub_df['pfx_z']*12,
            mode='markers',
            marker=dict(size=8, color=pitch_colors[pitch], opacity=0.4),
            name=pitch,
            showlegend=False,
            legendgroup=pitch
        ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=8, color=pitch_colors[pitch], opacity=1.0),
            name=pitch,
            showlegend=True,
            legendgroup=pitch
        ))

        fig.update_layout(
            width=350,
            height=345,
            xaxis=dict(range=[-27.5, 27.5],
                    scaleanchor='y',  
                    scaleratio=1,
                    showgrid=True,
                    zeroline=True,
                    gridcolor='lightgray',
                    zerolinecolor='black',
                    ),
            yaxis=dict(range=[-27.5, 27.5],
                       showgrid=True,
                       zeroline=True,
                       gridcolor='lightgray',
                       zerolinecolor='black',
                       ),
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            dragmode='select',
            legend=dict(
                orientation='h',  
                yanchor='bottom',
                y=0.01,  
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.6)',
                itemwidth=30,
            ),
            plot_bgcolor='white',  
            paper_bgcolor='white',
    )

    fig.update_traces(hoverinfo='skip')
    
    return fig

@app.callback(
    Output("popup", "style"),
    Input("open-popup", "n_clicks"),
    Input("close-popup", "n_clicks"),
    prevent_initial_call=True
)
def toggle_popup(open_clicks, close_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "open-popup":
        return {
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "rgba(0, 0, 0, 0.5)",
            "display": "flex", 
            "alignItems": "center",
            "justifyContent": "center",
            "zIndex": 1000
        }
    else:
        return {"display": "none"}

@app.callback(
    Output('count-checklist', 'value'),
    Input('all-counts-button', 'n_clicks'),
    Input('no-counts-button', 'n_clicks'),
    Input('pahead-counts-button', 'n_clicks'),
    Input('bahead-counts-button', 'n_clicks'),
    Input('even-counts-button', 'n_clicks'),
    Input('twok-counts-button', 'n_clicks'),
    Input('threeb-counts-button', 'n_clicks')
)
def set_preset(n1, n2, n3, n4, n5, n6, n7):
    if ctx.triggered_id == 'all-counts-button':
        return ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
    elif ctx.triggered_id == 'no-counts-button':
        return []
    elif ctx.triggered_id == 'pahead-counts-button':
        return ["0-1", "0-2", "1-2"]
    elif ctx.triggered_id == 'bahead-counts-button':
        return ["1-0", "2-0", "2-1", "3-0", "3-1"]
    elif ctx.triggered_id == 'even-counts-button':
        return ["0-0", "1-1", "2-2"]
    elif ctx.triggered_id == 'twok-counts-button':
        return ["0-2", "1-2", "2-2", "3-2"]
    elif ctx.triggered_id == 'threeb-counts-button':
        return ["3-0", "3-1", "3-2"]
    return dash.no_update

@app.callback(
    Output('applied-filters', 'data'),
    Input('close-popup', 'n_clicks'),
    State('pitch-type-checklist', 'value'),
    State('count-checklist', 'value'),
    prevent_initial_call=False  
)
def apply_filters(n_clicks, pitch_types, counts):

    return {
        'pitch_types': pitch_types,
        'counts': counts
    }


if __name__ == "__main__":
    app.run(debug=True)