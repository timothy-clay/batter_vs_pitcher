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
from flask_caching import Cache
import uuid
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

cache = Cache(server, config={"CACHE_TYPE": "SimpleCache"})

app.layout = html.Div([
    dcc.Store(id='data-store'),
    dcc.Store(id='applied-filters'),

    html.Div(id='input-screen', 
             className='visible-page',
             children=[
                html.H1("MLB Matchup Simulator"),
                html.P("Created by Timothy Clay"),
                html.Button("About This Project", id="about-page"),
                html.Div(
                    id="about-popup",
                    children=[
                        html.Div([
                            html.H2("About This Project", id="about-header"),

                            # Close button
                            html.Div([
                                html.Button("X", id="close-about", n_clicks=0, style={
                                    "background": "white", "color": "gray",
                                    "border": "none", "padding": "8px 12px",
                                    "borderRadius": "10px", "cursor": "pointer",
                                    "fontWeight": "bold", "fontSize":"15px"
                                })
                            ], style={"position": "absolute", "top": "25px", "right": "25px", "background": "transparent", "border": "none", "fontSize": "20px", "fontWeight": "bold", "cursor": "pointer", "color": "#333"}),

                            html.P("This site is an interactive summary dashboard that I created and published for my MLB matchup simulation project. I worked on and completed this personal project in its entirely in my free time during my 2025 summer internship with the Washington Nationals. A more thorough write-up of my methodology is included in the sections below. "),
                            html.P('To use the dashboard, enter a pitcher and batter using the dropdowns on the main page, then click the “Run Models” button. Once the models have run (which may take up to 30 seconds), it will load a new page that shows the simulation outcomes, allowing you to filter by pitch type, count, and pitch location. '),
                            html.P("With any questions, please feel free to reach me at clay.t@northeastern.edu. I hope you find this project as interesting as I did!"),
                            html.H3("Background / Abstract"),
                            html.P("The goal behind this project was to understand how different types of batters would fare against different types of pitchers over hundreds of plate appearances. I approached this projected with a Monte Carlo simulation, which generates thousands of pitches using realistic outcome probabilities, as produced by five feed-forward neural networks. Each of the five models considers a different aspect of the pitch, with models for pitch type, pitch location, swing decision, contact result, and batted ball outcome. To simulate a full plate appearance, I simply simulate individual pitches until an end condition (walk/strikeout/ball in play) is reached. Repeat this process for hundreds or thousands of iterations, and you can start to understand the long-term trends that may begin to appear. In my dashboard, I show these long-term trends, highlighting average pitch characteristics and usage rates, plate discipline and contact rates, per-600-PA counting stats, and more. "),
                            html.H3("Modeling Methodology"),
                            html.P("The heart of this entire project boils down to the five feed-forward neural networks that I use to predict individual aspects of a pitch. Two of the five predict what I would call pitcher-level features: the pitch type and its location. The remaining three models predict what I would call hitter-level features: the batter’s swing decision, contact result (if applicable), and batted ball outcome (if applicable). The models are all architecturally alike, which allowed for consistency across model training and evaluation. The models, however, do not all have the same inputs, which I discuss further below. "),
                            html.H4("Model Inputs"),
                            html.P('As each of the five models occurs predicts a “decision” that happens at a different point in the process of a pitch (e.g., a pitch type needs to be chosen before the batter decides whether to swing), the models cannot have the same exact inputs, though there is considerable overlap. Each of the five models share 21 inputs, with four of the five sharing 30: '),
                            html.Ul([
                                html.Li("4 capture situational context (count and handedness)"),
                                html.Li("8 capture batter swing characteristics (average swing speed, average swing length, etc.)"),
                                html.Li("9 capture batter plate discipline (swing and contact rates, hard-hit rate, etc.)"),
                                html.Li("9 capture pitch characteristics (velocity, spin, etc.) (NOTE: the pitch type model does not take these as inputs)"),
                            ]),
                            html.P("For the pitch type and pitch location models, they both also share 18 additional inputs, which all deal with the size of the pitcher’s arsenal. 17 of those 18 inputs are binary encodings indicating whether the pitcher has each possible pitch type in their arsenal, with the last input being a numeric count of how many total types of pitches are in the pitcher’s arsenal. Both models also take as input the ID of the pitcher. This input ID is then converted into a 4-dimensional encoding, which is combined with the rest of the data before training. The pitch location model also takes as input the chosen pitch type, which is then converted to an additional 8-dimensional encoding. The three hitter-level models, meanwhile, only have two unique inputs: the x and z coordinates of the pitch. "),
                            html.P("To process each of these inputs to be model-readable, I one-hot encoded all categorical variables and normalized all inputs to remove any biases related to unit size. I also implemented Bayesian shrinking for the plate discipline stats to prevent any players with small sample sizes from having misleading and outlier rate stats (such as a 100% swing rate)."),
                            html.H4("Model Architectures"),
                            html.P("To maintain consistency across each of my neural networks, I decided to standardize the architecture across all models. Each model was a single feed-forward neural network with 3 hidden layers. The first hidden layer was 128 dimensions, the second hidden layer was 64 dimensions, and the third hidden layer was only 32 dimensions. The first two hidden layers both included batch normalization and 10% dropout for regularization, while the final hidden layer only applies a ReLU activation before the output layer."),
                            html.P("Each of the five models also all work by outputting class probabilities (as opposed to a continuous value). For four of the five models, this makes logical sense; pitch type is clearly a multi-class categorical variable, and each of swing decision, contact result, and batted ball outcome can all be thought of as binary classification problems. For pitch location, however, intuition would suggest predicting a coordinate, rather than a class. I originally did construct the pitch location model this way, but I found that this implementation led to predictions that converged towards the center of the strike zone. This in turn resulted in inflated zone rates and deflated walk rates. To address this issue, I instead converted the pitch location into a categorical feature, dividing the potential pitch locations into a 10x10 grid with 100 different buckets, ranging from -2 feet to +2 feet on the x-axis and from 0 to 5 feet on the y-axis. While this approach sacrifices some granularity, the tradeoff for increased accuracy was well worth it. I may revisit this approach in the future to consider alternative ways to predict pitch location as a set of continuous variables, but for the time being, I’m satisfied with this approach."),
                            html.H4("Model Evaluation"),
                            html.P("To train my models, I used pitch-by-pitch data for every MLB game between the 2023 and 2025 All-Star breaks. Due to the differences in their inputs and outputs, each model had to be trained with a slightly different dataframe. The sizes of these dataframes are as follows:"),
                            html.Ul([
                                html.Li("Pitch Type Model: 1,427,534 observations"),
                                html.Li("Pitch Location Model: 1,412,234 observations"),
                                html.Li("Swing Decision Model: 1,415,269 observations"),
                                html.Li("Contact Result Model: 677,297 observations"),
                                html.Li("Batted Ball Outcome Model: 478,198 observations"),
                            ]),
                            html.P("Before training each model, I split the respective dataframe into training and testing splits. Each training split was 80% of the total observations, with the remaining 20% being used for model evaluation. During training, I evaluated each model using log loss. I chose this error metric because it successfully balanced rewarding the model for being confidently correct while penalizing it for being confidently incorrect. I was able to use the same error metric across all models because of the previously mentioned design decision to make each model predict class probabilities. "),
                            html.P("I trained each model for a maximum of 50 epochs. For each epoch, the model would learn the training data in batches of 64 observations and then predict on batches of testing data. If the log loss among the testing data did not improve in 10 epochs, I implemented early stopping to prevent overfitting. The final validation log loss for each of my models is shown below:"),
                            html.Ul([
                                html.Li("Pitch Type Model (17 classes): 1.248"),
                                html.Li("Pitch Location Model (100 classes): 3.696"),
                                html.Li("Swing Decision Model (2 classes): 0.425"),
                                html.Li("Contact Result Model (2 classes): 0.452"),
                                html.Li("Batted Ball Outcome Model (9 classes): 1.457"),
                            ]),
                            html.H3("Simulation Logic"),
                            html.P("With these five models, I was then able to simulate realistic outcomes for individual pitches. To sample a single pitch, I first pulled all the relevant information for the specified batter and the specified pitcher. This included information such as the batter’s plate discipline trends and the pitcher’s arsenal. This combined row of data contained all necessary information to be then passed into each subsequent model. "),
                            html.P("The first model in the pitch-simulation assembly line is the pitch type model. Intuitively, this model outputs a set of probabilities that each possible pitch type will be throw. The simulation loop then randomly samples from those outputs (using the probabilities as class weights). This pitch is then added to the row of data, and the simulation continues. "),
                            html.P("The next model the simulation uses is the pitch location model. Before passing the data to the pitch location model itself, the simulation loop fills in the relevant pitch characteristics based on the pitch type that was previously chosen. It does this by sampling a value for each pitch characteristic using that characteristic’s mean and standard deviation. With these added pitch characteristics, the simulation loop then passes the row into the pitch location model to predict the probability that the pitch will be thrown in each of the 100 pitch location buckets. The loop then samples from those buckets using their class probabilities, then chooses the specific point within that bucket by taking a random point within the bucket’s bounds. This location coordinate is then added to the row, and the simulation continues. "),
                            html.P("The final step(s) of the pitch-level simulation call the three hitter-level models. The simulation loop first gets the probability of a swing using the swing decision model, then samples from those probabilities whether the batter swung. If the sampled swing decision was not a swing, then the simulation end. Otherwise, the simulation continues by predicting and sampling whether the batter makes contact. Again, if the sample contact result did not result in contact, the simulation ends. If the simulation loop predicts that the batter will both swing and make contact, however, the last step in the process is to predict the outcome of that contact. As with all the other models, this model predicts the probability of each batted ball outcome, including fouls, and samples from those probabilities to choose the result of the swing. "),
                            html.P("When the pitch-level simulation has finished, the model returns the row. When multiple pitches are being simulated in succession, these rows can be appended together as a dataframe. "),
                            html.P("To simulate plate appearances, the model continues to simulate individual pitches, adding to a cumulative dataframe of pitch results, until a stop condition is met. A stop condition only happens when the batter walks (takes a ball with 3 balls), strikes out (takes or swings at a strike with 2 strikes) or puts the ball into play in fair territory. After each pitch, the plate appearance simulation checks these conditions and only stops if one is met. When a stop condition is reached, the simulation returns the cumulative dataframe containing the outcomes of each individual pitch. This plate-appearance-level simulation can iterate for any specified number of times and can continue to add upon one large cumulative result dataframe. Throughout my project, I defaulted to simulating 1,000 plate appearances for each match-up.  "),
                            html.H3("Summary Dashboard"),
                            html.P("The last step of my project (for now!) was to make the results of my simulations easy to access and explore. To do this, I developed this very dashboard using Plotly Dash in Python. This was my first large-scale project using the framework, so there was a bit of a learning curve, but I was pleased by the level of control I had over the site’s functionality and appearance, and it was exceptionally easy to integrate the Plotly visualizations I made throughout the process. "),
                            html.P("Hosting this site on the web was also a new experience, as it required me to launch and manage an AWS EC2 instance and connect said instance to my domain name (app.timothyclay.dev). When finalizing the site, I had to make some tweaks to the code for performance and compatibility reasons. For instance, instead of storing the results of my simulations in local memory, I incorporated Redis to cache the data. Despite this tweak, I was still slightly disappointed in the spike in runtime transitioning from my local machine to the EC2 instance. In the future, I may look to try to parallelize the plate appearance simulations to decrease the overall runtime.  "),
                            html.H3("Future Work"),
                            html.P("Beyond the work I’ve already done, I’m excited to continue exploring how I may be able to turn the results of my simulations into more usable and actionable tools. I have a variety of potential ideas for future applications of this work, including substitution and lineup optimization.   "),
                            html.P("For substitution optimization, I believe that these simulation models could be valuable tools to help decide which relief pitchers or pinch hitters to bring in. In both cases, understanding which available players are likely to see the most success over the long run help inform the decision of which player to use. These tools should be fairly easy to implement, and I’m looking forward to creating them soon. "),
                            html.P("Another potential downstream use case for this project is lineup optimization. While my simulation is currently base-out-state-agnostic, I could very easily tweak the simulation logic to allow for bases and outs to be recorded. In doing so, it would be possible to simulate the production of an entire lineup against a given pitcher. This could help teams optimize their lineup construction by choosing the lineup that is expected to produce the most runs."),
                            html.P("Both ideas are what I believe to be the tip of the iceberg when it comes to future applications of this project. Having a way to simulate the expected outcomes of any pitcher/batter match-up will be incredibly valuable and will open tons of doors for future research. As I continue to refine this project, I’m excited to continue exploring possible applications of this work, and I will be sure to share any subsequent projects I tackle."),

                            html.Div([
                                html.Button("← Return to Dashboard", id="close-about-2", n_clicks=0, style={
                                    "background": "#349eeb", "color": "white",
                                    "border": "none", "padding": "8px 16px",
                                    "borderRadius": "6px", "cursor": "pointer", "marginTop":"20px",
                                    "fontWeight": "bold", "fontSize":"15px"
                                })
                            ], style={"background": "transparent", "border": "none", "fontSize": "20px", "fontWeight": "bold", "cursor": "pointer", "color": "#333"}),

                        ], className="modal-box", style={
                            "background": "white",
                            "padding": "25px",
                            "borderRadius": "10px",
                            "boxShadow": "0 4px 20px rgba(0,0,0,0.2)",
                            "width": "70%",
                            "maxWidth": "800px",
                            "maxHeight": "80vh",
                            "overflowY": "auto",
                            "position":"relative"
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
                html.Div(
                    id='input-screen-container',
                    children=[

                        # Left: Pitcher headshot
                        html.Div(
                            id="input-pitcher-headshot",
                        ),

                        # Middle: your original input-screen box
                        html.Div(
                            id="dropdowns-section",
                            children=[
                                html.H2("Choose Matchup"),
                                html.Label("Pitcher"),
                                dcc.Dropdown(
                                    id='pitcher-dropdown',
                                    options=pitcher_options,
                                    placeholder='Select a pitcher'
                                ),

                                html.Label("Batter"),
                                dcc.Dropdown(
                                    id='batter-dropdown',
                                    options=batter_options,
                                    placeholder='Select a batter'
                                ),

                                dcc.Loading(
                                    children=[
                                        html.Div(
                                            id='loading-overlay-output',
                                            children=[
                                            html.Button(
                                                'Run Model',
                                                id='run-model',
                                                n_clicks=0
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
                            id="input-batter-headshot"
                        )
                    ]),
            ]),

    
    
    html.Div(id='output-screen', className='hidden-page', children=[

        html.Div(id='left-output-screen',
                 children=[
            html.Div(
                id='matchup-id-section',
                children=[
                    html.Div(id='pitcher-headshot-container', style={'flex': '0 0 auto'}),
                    html.Div(
                        id='matchup-text',
                        children=[
                            html.Div(id="pitcher-name-container"),
                            html.P("vs."),
                            html.Div(id="batter-name-container")
                        ]
                    ),
                    html.Div(id='batter-headshot-container', style={'flex': '0 0 auto'})
                ]
            ),
            html.Div(id='summary-output'),
        ]), 
        html.Div(
            id='right-output-screen', children=[
            html.Div(id="summary-table"),
            html.Div(
                id='pitch-plots',
                     children=[
                         html.Div([
                             html.H3('Pitch Break'),
                             dcc.Graph(id='pitch-break-plot', 
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
        html.Button("⚙ Filters", id="open-popup"),
        html.Div(
            id="popup",
            children=[
                html.Div([
                    html.H2("Filters", id="filters-header"),

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
    Output('input-screen', 'className'),
    Output('output-screen', 'className'),
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

    input_style = 'visible-page'
    output_style = 'hidden-page'

    if button_id == 'back-button':
        return None, dash.no_update, input_style, output_style

    if batter is None or pitcher is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    df = sample_pas(batter, pitcher, models=models, data=data)

    # Create a unique key and cache it server-side
    key = str(uuid.uuid4())
    cache.set(key, df, timeout=600)  # store for 10 minutes

    return key, dash.no_update, 'hidden-page', 'visible-page'

@app.callback(
    Output('summary-output', 'children'),
    Input('scatter-plot', 'selectedData'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_summary(selectedData, stored_data, applied_filters):

    if stored_data is None:
        return html.Div("No data to summarize.")

    df = cache.get(stored_data)

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
    

    return html.Div(
        id='result-tables',
        children=[
            html.H3("Summary Data"),
            html.Div(
                className="data-table",
                children=[
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
                ]
            ),
            
            html.H3("Rate Stats"),
            html.Div(
                className="data-table",
                children=[
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
                    )
                ]
            ),
            
            html.H3("Stats per 600 PA"),
            html.Div(
                className='data-table',
                children=[
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
                    )
                ]
            ),
            
            html.H3("Plate Discipline"),
            html.Div(
                className='data-table',
                children=[
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
                ]
            )
    ]
)

@app.callback(
    Output('summary-table', 'children'),
    Input('scatter-plot', 'selectedData'),
    Input('data-store', 'data'),
    Input('applied-filters', 'data')
)
def update_pitches_summary_table(selectedData, stored_data, applied_filters):
    if stored_data is None:
        return html.Div("No data to summarize.")

    df = cache.get(stored_data)

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

    return html.Div(
        id='pitch-table',
        children=[
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
            )
        ]
    )

@app.callback(
    Output('pitch-type-checklist', 'options'),
    Output('pitch-type-checklist', 'value'), 
    Input('data-store', 'data')
)
def update_pitch_type_checklist(stored_data):

    if stored_data is None:
        return [], []

    df = cache.get(stored_data)
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
        id="pitcher-headshot",
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pitcher_id}/headshot/67/current.png'
    ) if pitcher_id else ""

    pitcher_name = html.H2(
        pitcher_dict[pitcher_id],
        style={'margin': '0'}
    ) if pitcher_id else ""
    
    batter_img = html.Img(
        id="batter-headshot",
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{batter_id}/headshot/67/current.png'
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
        id='pitcher-dropdown-headshot',
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pitcher_id}/headshot/67/current.png',
    ) if pitcher_id else ""
    
    return pitcher_img

@app.callback(
    Output('input-batter-headshot', 'children'),
    Input('batter-dropdown', 'value')
)
def update_input_batter_headshots(batter_id):
    batter_img = html.Img(
        id='batter-dropdown-headshot',
        src=f'https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{batter_id}/headshot/67/current.png'
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
    
    df = cache.get(stored_data)

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
    
    df = cache.get(stored_data)

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

@app.callback(
    Output("about-popup", "style"),
    Input("about-page", "n_clicks"),
    Input("close-about", "n_clicks"),
    Input("close-about-2", "n_clicks"),
    prevent_initial_call=True
)
def manage_about_page(open_clicks, close_clicks, close_clicks_2):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "about-page":
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


if __name__ == "__main__":
    app.run(debug=True)