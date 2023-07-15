import matplotlib.pyplot as plt
import numpy as np
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
from nba_api.stats.static import players
import json
import requests
import pandas as pd
import time
import streamlit as st
import plotly.graph_objects as go
from copy import deepcopy

# seasons_list = ['2022-23']
# nba_teams = teams.get_teams()
# nba_players = players.get_players()
# shots_data = []

# GET SHOTS DATA FOR ALL NBA TEAMS FOR THE 2022-2023 NBA SEASON. CREATE A CSV FILE CALLED shots_data.csv. CLEAN AND AGGREGATE THE DATA IN SQL

# def get_all_shots(seasons, teams):
#     for year in seasons:
#         for nba_team in teams:
#             try:
#                 team_shots_json = shotchartdetail.ShotChartDetail(player_id=0, team_id=nba_team['id'], context_measure_simple='FGA', season_nullable=year, season_type_all_star='Regular Season', timeout=150)
#                 team_shots_data = json.loads(team_shots_json.get_json())
#                 rows = team_shots_data['resultSets'][0]['rowSet']
#                 headers = team_shots_data['resultSets'][0]['headers']
#                 team_shots_df = pd.DataFrame(rows)
#                 team_shots_df.columns = headers
#                 team_shots_df.insert(0, "SEASON", year)
#                 # team_shots_df = team_shots_df[['SEASON', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_NAME', 'ACTION_TYPE', 'SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']]
#                 shots_data.append(team_shots_df)
#                 print(f"Getting shots for {nba_team['full_name']} for the {year} season")
#                 time.sleep(1)
#             except ValueError:
#                 print(f"Value error for {nba_team['full_name']} for the {year} season")
#
# get_all_shots(seasons_list, nba_teams)
# season_shots_df = pd.concat(shots_data)
# season_shots_df.reset_index(drop=True, inplace=True)
# season_shots_df.to_csv('shots_data.csv')


# DRAW PLOTLY COURT FUNCTION: https://towardsdatascience.com/interactive-basketball-data-visualizations-with-plotly-8c6916aaa59e
def draw_plotly_court(fig, fig_width=600, margins=10):
    import numpy as np

    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200,
                    closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ),

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ),
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ),

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return True

# PLOT DATA USING PLOTLY
def create_plotly_shot_chart(selection, team_or_player):

    # OPEN SHOT DATA FROM NBA_API
    pd.set_option('display.max_columns', None)
    all_shots_df = pd.read_csv('shots_data.csv')
    all_shots_df = all_shots_df[['PLAYER_NAME', 'TEAM_NAME', 'SHOT_TYPE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']]

    # USE MATPLOTLIB TO GET BIN INFORMATION FROM THE SHOT DATA

    all_shots_plot = plt.hexbin(
        x=all_shots_df[all_shots_df[f'{team_or_player}_NAME'] == selection]['LOC_X'],
        y=all_shots_df[all_shots_df[f'{team_or_player}_NAME'] == selection]['LOC_Y'],
        extent=(-250, 250, -47.5, 422.5),  # x and y limit of the plot
        cmap='Blues',
        gridsize=51)

    plt.close()  # closes the plot window so we can continue and get more information

    fgs_made = all_shots_df[(all_shots_df['SHOT_MADE_FLAG'] == 1) & (all_shots_df[f'{team_or_player}_NAME'] == selection)]

    fgs_made_plot = plt.hexbin(
        x=fgs_made['LOC_X'],
        y=fgs_made['LOC_Y'],
        extent=(-250, 250, -47.5, 422.5),
        cmap='Reds',
        gridsize=51
    )
    plt.close()

    # all league shots, will be used to get all shots per hexbin for entire league
    league_all_shots_plot = plt.hexbin(
        x=all_shots_df['LOC_X'],
        y=all_shots_df['LOC_Y'],
        extent=(-250, 250, -47.5, 422.5),
        gridsize=51
    )
    plt.close()

    league_fgs_made_plot = plt.hexbin(
        x=all_shots_df[all_shots_df['SHOT_MADE_FLAG'] == 1]['LOC_X'],
        y=all_shots_df[all_shots_df['SHOT_MADE_FLAG'] == 1]['LOC_Y'],
        extent=(-250, 250, -47.5, 422.5),
        gridsize=51
    )
    plt.close()

    # NOTES: .get_array() gets the number of shots for each hexbin (total number of shots made, total number of shots attempted for each bin)
    fg_accuracy_per_bin = fgs_made_plot.get_array() / all_shots_plot.get_array()  # FG accuracy for each bin, outputs an np masked array
    fg_accuracy_per_bin[np.isnan(
        fg_accuracy_per_bin)] = 0  # Makes NaN values 0 (bins that have no shot attempts are divided by 0, results in NaN)

    # same as above but for the entire league
    league_fg_accuracy_per_bin = league_fgs_made_plot.get_array() / league_all_shots_plot.get_array()
    league_fg_accuracy_per_bin[np.isnan(league_fg_accuracy_per_bin)] = 0

    x = [i[0] for i in all_shots_plot.get_offsets()]  # gets the x coord in tuple for each bin appends to list x
    y = [i[1] for i in all_shots_plot.get_offsets()]  # gets the y coord in tuple for each bin appends to list y

    # OPEN FILE WITH COORDINATES AND ZONE INFORMATION TO SMOOTH OUT ACCURACY ON THE PLOT
    zones_df = pd.read_csv('zones.csv')

    # CREATE A DF USING THE INFORMATION FROM MATPLOTLIB
    df_plotly_data = pd.DataFrame({'x_loc': x, 'y_loc': y,
                                   'made_shots_per_bin': fgs_made_plot.get_array(),
                                   'shots_per_bin': all_shots_plot.get_array(),
                                   'fg_acc_per_bin': fg_accuracy_per_bin,
                                   'bin_frequency': all_shots_plot.get_array() / sum(all_shots_plot.get_array()),
                                   'league_made_shots_per_bin': league_fgs_made_plot.get_array(),
                                   'league_shots_per_bin': league_all_shots_plot.get_array(),
                                   'league_fg_acc_per_bin': league_fg_accuracy_per_bin,
                                   'league_bin_frequency': league_all_shots_plot.get_array() / sum(
                                       league_all_shots_plot.get_array())
                                   })

    # ADD IN ZONE INFORMATION ON THE ABOVE DF, CHANGE FG ACCURACY TO BE ZONE FG ACCURACY INSTEAD OF BIN ACCURACY
    df_plotly_data = df_plotly_data.fillna(0)
    df_plotly_data = df_plotly_data.sort_values(by=['x_loc', 'y_loc'], ascending=[True, False]).reset_index(drop=True)
    zones_df = zones_df.sort_values(by=['xlocs', 'ylocs'], ascending=[True, False]).reset_index(drop=True)
    df_plotly_data = pd.concat([df_plotly_data, zones_df['shot_zone']], axis=1)
    zone_list = list(df_plotly_data['shot_zone'].unique())
    group_data = df_plotly_data.groupby('shot_zone')
    df_plotly_data['made_shots_per_bin_total'] = group_data['made_shots_per_bin'].transform('sum')
    df_plotly_data['shots_per_bin_total'] = group_data['shots_per_bin'].transform('sum')
    df_plotly_data['league_made_shots_per_bin_total'] = group_data['league_made_shots_per_bin'].transform('sum')
    df_plotly_data['league_shots_per_bin_total'] = group_data['league_shots_per_bin'].transform('sum')
    df_plotly_data['zone_fg_accuracy'] = df_plotly_data['made_shots_per_bin_total'] / df_plotly_data[
        'shots_per_bin_total']
    df_plotly_data['league_zone_fg_accuracy'] = df_plotly_data['league_made_shots_per_bin_total'] / df_plotly_data[
        'league_shots_per_bin_total']
    df_plotly_data = df_plotly_data.fillna(0)
    df_plotly_data['fg_acc_per_bin'] = df_plotly_data['zone_fg_accuracy']
    df_plotly_data['league_fg_acc_per_bin'] = df_plotly_data['league_zone_fg_accuracy']

    if team_or_player == 'TEAM':
        # IF TEAM, MINIMUM TO BE SHOWN ON PLOT IS 0.05 OF TOTAL SHOTS
        min_frequency_threshold = 0.0005  # minimum pct of shots frequency to be shown on plot
        df_plotly_data.loc[df_plotly_data['bin_frequency'] < min_frequency_threshold, 'bin_frequency'] = 0
    else:
        # IF PLAYER, MINIMUM TO BE SHOWN ON PLOT IS 3 SHOTS ON BIN
        min_frequency_threshold = 3
        df_plotly_data.loc[df_plotly_data['shots_per_bin'] < min_frequency_threshold,'shots_per_bin'] = 0

    # CREATE A TEMP COPY OF THE DATA, USE THE DATA TO GET THE RELATIVE FG% BETWEEN SELECTION AND THE ENTIRE LEAGUE
    rel_hexbin_stats = deepcopy(df_plotly_data)
    base_hexbin_stats = deepcopy(df_plotly_data)
    rel_hexbin_stats['fg_acc_per_bin'] = rel_hexbin_stats['fg_acc_per_bin'] - base_hexbin_stats['league_fg_acc_per_bin']

    # WILL BE USED FOR THE SIZE, FREQUENCY PER HEXBIN IS LIMITED TO A MAX OF 3.5% TO KEEP SIZE SCALES IN CHECK
    max_frequency = 0.0035
    hex_frequency_input = np.array([min(max_frequency, i) for i in rel_hexbin_stats['bin_frequency']])

    # COLORSCALE FOR COLORBAR, CLIPPING THE MARKERS TO 5% WORSE OR 5% BETTER
    colorscale = 'Portland'
    marker_cmin = -0.05
    marker_cmax = 0.05
    # ticktexts = [str(marker_cmin*100)+'%-', "", str(marker_cmax*100)+'%+']
    ticktexts = ['Worse', 'Average', 'Better']

    hexbin_text = [
        '<i>Accuracy: </i>' + str(round(df_plotly_data['fg_acc_per_bin'][i] * 100, 1)) + '%<BR>'
                                                                                         '<i>League Accuracy: </i>' + str(
            round(df_plotly_data['league_fg_acc_per_bin'][i] * 100, 1)) + '%<BR>'
                                                                          '<i>Frequency: </i>' + str(
            round(df_plotly_data['bin_frequency'][i] * 100, 2)) + '%'
        for i in range(len(hex_frequency_input))
    ]

    fig = go.Figure()
    draw_plotly_court(fig)
    fig.add_trace(go.Scatter(
        x=rel_hexbin_stats['x_loc'],
        y=rel_hexbin_stats['y_loc'],
        mode='markers',
        name='markers',
        text=hexbin_text,
        marker=dict(
            size=hex_frequency_input,
            sizemode='area',
            sizeref=2. * max(hex_frequency_input) / (12. ** 2),
            sizemin=2.5,
            color=rel_hexbin_stats['fg_acc_per_bin'],
            colorscale=colorscale,
            colorbar=dict(
                thickness=15,
                x=0.84,
                y=0.87,
                yanchor='middle',
                len=0.2,
                title=dict(
                    text="FG%",
                    font=dict(size=11, color='#4d4d4d'),
                ),
                tickvals=[marker_cmin, (marker_cmin + marker_cmax) / 2, marker_cmax],
                ticktext=ticktexts,
                tickfont=dict(size=11, color='#4d4d4d'),
            ),
            cmin=marker_cmin,
            cmax=marker_cmax,
            line=dict(width=1, color='#333333'),
            symbol='hexagon',
        ), hoverinfo='text'
    ))
    return fig



shots_aggregate_player = pd.read_csv('aggregate_data.csv')
shots_aggregate_team = pd.read_csv('aggregate_data_team.csv')
player_list = shots_aggregate_player['player_name'].unique()
team_list = shots_aggregate_player['team_name'].unique()


st.set_page_config(
        page_title="NBA Shot Charts Sample Project", layout='wide'
)

st.markdown("## NBA Shot Charts Sample Project")


tab1, tab2, tab3 = st.tabs(['Team Shot Charts', 'Player Shot Charts', 'Info'])

with tab1:
    st.markdown("### Team Shot Charts Using Plotly (22-23 NBA Regular Season)")
    st.markdown("###### Credit to this article from JP Hwang for the court drawing, how to plot the data using Plotly, "
                "and various other things: "
                "https://towardsdatascience.com/interactive-basketball-data-visualizations-with-plotly-8c6916aaa59e")

    if 'selectboxes' not in st.session_state:
        st.session_state.selectboxes = 1

    # Create a button to add a new select box
    if st.session_state.selectboxes < 2 and st.button('Add Team'):
        st.session_state.selectboxes +=1

    columns = {'col1': '', 'col2': ''}
    columns['col1'], columns['col2'] = st.columns(2)

    # Create select boxes
    for i in range(st.session_state.selectboxes):
        with columns[f'col{i+1}']:
            selected_option = st.selectbox(f'Select Team {i + 1}', team_list)
            # Generate Plotly plot based on selected option
            plotly_fig = create_plotly_shot_chart(selected_option, 'TEAM')
            st.plotly_chart(plotly_fig, use_container_width=True)

with tab2:
    st.markdown("### Player Shot Charts Using Plotly (22-23 NBA Regular Season)")
    st.markdown("###### Credit to this article from JP Hwang for the court drawing, how to plot the data using Plotly, "
                "and various other things: "
                "https://towardsdatascience.com/interactive-basketball-data-visualizations-with-plotly-8c6916aaa59e")

    if 'selectboxes_player' not in st.session_state:
        st.session_state.selectboxes_player = 1

    # Create a button to add a new select box
    if st.session_state.selectboxes_player < 2 and st.button('Add Player'):
        st.session_state.selectboxes_player +=1

    columns_player = {'col1': '', 'col2': ''}
    columns_player['col1'], columns_player['col2'] = st.columns(2)

    # Create select boxes
    for i in range(st.session_state.selectboxes_player):
        with columns_player[f'col{i+1}']:
            selected_option = st.selectbox(f'Select Player {i + 1}', player_list)
            # Generate Plotly plot based on selected option
            plotly_fig = create_plotly_shot_chart(selected_option, 'PLAYER')
            st.plotly_chart(plotly_fig, use_container_width=True)

with tab3:
    st.markdown("### Team Shot Charts Using Plotly (22-23 NBA Regular Season)")
    st.markdown("###### Credit to this article from JP Hwang for the court drawing, how to plot the data using Plotly, "
                "and various other things: "
                "https://towardsdatascience.com/interactive-basketball-data-visualizations-with-plotly-8c6916aaa59e")

    st.markdown("1. This project was created in an effort to show potential employers that I do have working and "
                "useable knowlege in Python, SQL, APIs, reading documentation, and cleaning and transforming data. "
                "I've never held a tech role and would like to transition into one, thus this sample project.")
    st.markdown("2. Most of the information on how to plot the information on Plotly, draw the court, and "
                "how to visualize NBA shot comes from the article linked above, huge credit to JP Hwang. I learned a"
                "lot just from reading his articles/blogs")
    st.markdown("3. The work that I did mostly comes from getting, cleaning, and transforming the data in order to"
                "plot and visualize it. The article was working with data from the 19-20 NBA season and the data from"
                "the article had already been summarized. What I needed to do was get data for the latest NBA season,"
                "summarize, aggregate, and transform that data depending on different conditions"
                " to be able to plot the shot charts. Basically be able to reverse engineer what I had available and"
                "fit that into the Plotly")







