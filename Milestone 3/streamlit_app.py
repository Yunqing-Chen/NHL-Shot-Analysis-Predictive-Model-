import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import re
import os
import json
import requests

os.sys.path.append((pathlib.Path(__file__) / '..' ).resolve().absolute())
#from serving import app


"""
General template for your streamlit app.
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("HOCKEY VIZ TOOL")

URL = 'http://0.0.0.0:8080'


st.sidebar.header("Select model")
model_selection_form = st.sidebar.form('Model')
model_selection_form.write('Default entity: IFT6758_2024-B01')
workspace = model_selection_form.text_input('Workspace:', placeholder='ms2-logistic-regression')
model = model_selection_form.text_input('Model:')
model_version = model_selection_form.text_input('Version:')
model_submit = model_selection_form.form_submit_button('Submit')
if model_submit and (model == '' or model_version == '' or workspace == ''):
    model_selection_form.warning('Empty field Workspace/Model/Version')
if model_submit and (model != '' and model_version != '' and workspace != ''):
    print(type(model))
    print(type(model_version))
    st.write('Fetching model', model, 'with version', model_version, 'in', workspace)
    model_selection_json = {
        'workspace': workspace,
        'model': model,
        'version': model_version
    }
    response = requests.post(f"{URL}/download_registry_model", json = model_selection_json)
    st.write(response.status_code, response.reason)
    if response.status_code != 200:
        st.write(response.content)
#with st.sidebar:
#    st.header("Select model")
#    with st.form('Model'):
#        model = st.text_input('Model:')
#        version = st.text_input('Version:')
#        submit = st.form_submit_button('Submit')
#        if submit:
#            st.write('Fetching game')

#list_downloaded_games = pd.read_csv(pathlib.Path('./ift6758/dataset/complex_engineered/augmented_data.csv'))['game_id'].unique().tolist()
@st.cache_data(max_entries=1)
def list_downloadable_games():
    gen_possible_games = [possible_gid for possible_gid in range(2015020001, 2024029999) if re.match('20[1-2][0-9]02[0-1][0-9]{3}', str(possible_gid))]
    list_downloadable_games = [gid for gid in gen_possible_games if int(str(gid)[-4:]) < 1350]
    return list_downloadable_games
#list_available_games = list_downloaded_games + list_downloadable_games

game_id_container = st.container()
list_downloadable_games = list_downloadable_games()
game_id_selectbox = game_id_container.selectbox('Game ID', list_downloadable_games)
game_id_submit = game_id_container.button('Ping game')
if game_id_submit:
    request = { 'game_id': game_id_selectbox }
    response = requests.post(URL+'/process_game', json=request)
#with st.container():
#    game_id = st.text('Game ID')
#    list_available_games = pd.read_csv(pathlib.Path('./ift6758/dataset/complex_engineered/augmented_data.csv'))['game_id'].unique()
#    game_id_entry = st.selectbox('Game ID', list_available_games)
#    submit_gid = st.button('Ping game')
#    if submit_gid:
#        request = { 'game_id': game_id_entry }
#        response = requests.post('http://0.0.0.0:8080/process_game', json=request)



predictions_container = st.container()
if not game_id_submit:
    st.write('Waiting for game input')
else:
    response_json = response.json()
    if response_json == '':
        predictions_container.error('Got empty response, is game valid ?')
    #st.write(response_json.get('message'))
    if not str(response.status_code).startswith('2'):
        with open('./serving/flask.log', 'r') as logs:
            tail_logs = ''
            for line in (logs.readlines() [-20:]):
                tail_logs += line+'\n'
            st.error('An error occured while generating predictions, please refer to logs\n\n')
            st.error(tail_logs)
    else:
        if "xg_df" not in st.session_state:
            df = pd.DataFrame(
              data=[ (game_event.get('predicted_probabilities'), game_event.get('team_id')) for game_event in response_json ],
              columns=['goal_proba', 'team_id']
            )
            home_game_xG = df.groupby(by='team_id').sum()
            home_game_xG
            st.session_state.xg_df = home_game_xG
            #away_game_xG·=·
        else:
            st.session_state.xg_df
#with st.container():
    # TODO: Add Game info and predictions
    #response_json = json.dumps(response)

    #df = pd.DataFrame(
    #  data=[ (game_event.get('predicted_probabilities'), game_event.get('team_id')) for game_event in response_json ],
    #  columns=['goal_proba', 'team_id']
    # )
    #home_game_xG = df.groupby(by='team_id').sum()
    #away_game_xG = 
#    st.write('home_team')
#    st.write('away_team')

with st.container():
    # TODO: Add data used for predictions
    with st.expander('Data'):
        pass
        #response.game_df
    with st.expander('Input features'):
        pass
        #response.game_df[['distance_from_net', 'angle_from_net']]
