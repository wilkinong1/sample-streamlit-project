import numpy as np
# from nba_api.stats.endpoints import shotchartdetail
# from nba_api.stats.static import teams
# from nba_api.stats.static import players
import json
import requests
import pandas as pd
import plotly.express as px
import time
import streamlit as st

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

# OPEN AGGREGATED DATA AND PLOT
shots_aggregate_all = pd.read_csv('aggregate_data_all.csv')
figure_1 = px.scatter(shots_aggregate_all[shots_aggregate_all['shot_attempted_flag'] > 50],
                      x='shot_distance', y='accuracy', size='shot_attempted_flag', color='shot_type', size_max=30)

shots_aggregate_player = pd.read_csv('aggregate_data.csv')
shots_aggregate_team = pd.read_csv('aggregate_data_team.csv')
player_list = shots_aggregate_player['player_name'].unique()
team_list = shots_aggregate_player['team_name'].unique()


def select_player(player_name):
    fig = px.scatter(shots_aggregate_player[(shots_aggregate_player['player_name'] == player_name) & (
                shots_aggregate_player['shot_distance'] <= 35)],
                     x='shot_distance', y='accuracy', size='shot_attempted_flag', color='shot_type', size_max=30)
    return fig

def select_team(team_name):
    fig = px.scatter(shots_aggregate_team[(shots_aggregate_team['team_name'] == team_name) & (
            shots_aggregate_team['shot_distance'] <= 35)],
                     x='shot_distance', y='accuracy', size='shot_attempted_flag', size_max=30, color='shot_type')
    return fig

st.markdown("## Sample Project")
st.markdown("#### League Accuracy for Each Shot Distance")
st.plotly_chart(figure_1)


st.markdown("#### Accuracy for Selected Player For Each Shot Distance")
player_selected = st.selectbox(label="Player Name", options=player_list, index=0)
st.plotly_chart(select_player(player_selected))

st.markdown("#### Accuracy for Selected Team For Each Shot Distance")
team_selected = st.selectbox(label='Team Name', options=team_list, index=0)
st.plotly_chart(select_team(team_selected))



