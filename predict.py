
import os
import random
import records
import tensorflow as tf
import pandas as pd
import numpy as np

team_start_marker = ''
team_end_marker = 'T'
wrestler_start_marker = 'x'
wrestler_end_marker = 'y'

def clean_text(text):
    import unicodedata
    temp_string = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode()
    temp_string = temp_string.lower()
    temp_string = temp_string.replace("'", "")
    temp_string = temp_string.replace('"', "")
    temp_string = temp_string.replace('.', "")
    return temp_string

def lookup(alias, db):
    assert isinstance(alias, str)
    cleaned_alias = clean_text(alias)
    id_dict = db.query("SELECT id FROM alias_table WHERE alias = '{alias}';".format(alias=cleaned_alias), fetchall=True).as_dict()
    if id_dict:
        id = id_dict.get('id')
    else:
        id = 0
        
    return id
    
def make_prediction_series(alias_list, db, series_type, event_date=29991231, number_of_history_matches=10, max_number_of_wrestlers=75):
    assert series_type in ['test', 'train', 'predict']
    wrestler_index_template_list = ['id', 'dob', 'nationality']
    history_match_index_template_list = ['days_since_match', 'won_match', 'wintype', 'title', 'matchtype', 'opponents', 'allies']
    index_list = []
    matches_query_string = """
SELECT m.date, m.wintype, m.title, m.matchtype, c.competitors
FROM match_table m JOIN competitor_table c ON m.id = c.id
WHERE date < {event} AND id IN (
    SELECT id as match_id FROM competitors_table WHERE competitors LIKE '%{start}{id}{end}%'
)
ORDER BY date DESC, LIMIT {limit}"""
    
    for wrestler_number in range(max_number_of_wrestlers):
        for wrestler_template in wrestler_index_template_list:
            index_list.extend("w{wn}_{temp}".format(wn=wrestler_number, temp=wrestler_template))
            for history_number in range(number_of_history_matches):
                for history_template in history_match_index_template_list:
                    index_list.extend("w{wn}m{mn}_{temp}".format(wm=wrestler_number, mn=match_number, temp=history_template))
    
    if series_type in ['test', 'train']:
        index_list.extend(['current_title', 'current_matchtype'])
        for wrestler_number in range(max_number_of_wrestlers):
            index_list.extend(['w{wn}_current_allies'.format(wn=wrestler_number), 'w{wn}_current_opponents'.format(wn=wrestler_number)])
    
    prediction_series = pd.Series(0, index=index_list)
    
    current_match = None        # prevents not-allocated failure, possibly change in future??
    for wrestler_number, alias in enumerate(alias_list):
        id = lookup(alias, db)
        stats_query = db.query("SELECT dob, nationality FROM wrestlers_table WHERE id = '{id}'".format(id=id)).as_dict()
        
        if series_type in ['test', 'train']:
            matches_query = db.query(matches_query_string.format(
                event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches+1
            )).as_dict()
            if not current_match:
                current_match = matches_query.pop()
                prediction_series['current_wintype'] = current_match['wintype']
                prediction_series['current_title'] = current_match['title']
                prediction_series['current_matchtype'] = current_match['matchtype']
        
            current_match_competitors_list = []
            for team in current_match['competitors'].split(team_end_marker):
                current_match_competitors_list.append(team[1,-1].strip(wrestler_start_marker).split(wrestler_end_marker))
            for team in current_match_competitors_list:
                if id in team:
                    prediction_series['w{wn}_current_allies'.format(wn=wrestler_number)] = len(team)-1
                else:
                    prediction_series['w{wn}_current_opponents'.format(wn=wrestler_number)] += len(team)
            
        else:
            matches_query = db.query(matches_query_string.format(
                event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches
            )).as_dict()
    
        prediction_series['w{wn}_id'.format(wn=wrestler_number)] = id
        prediction_series['w{wn}_dob'.format(wn=wrestler_number)] = stats_query.get('dob',0)
        prediction_series['w{wn}_nationality'.format(wn=wrestler_number)] = stats_query.get('nationality',0)
        for match_number, match in enumerate(matches_query):

        prediction_series['w{wn}m{mn}_days_since_match'.format(wn=wrestler_number, mn=match_number)] = event_date - match['date']
        prediction_series['w{wn}m{mn}_wintype'.format(wn=wrestler_number, mn=match_number)] = match['wintype']
        prediction_series['w{wn}m{mn}_title'.format(wn=wrestler_number, mn=match_number)] = match['title']
        prediction_series['w{wn}m{mn}_matchtype'.format(wn=wrestler_number, mn=match_number)] = match['matchtype']
        
        competitors_list = []
        for team in match['competitors'].split(team_end_marker):
            competitors_list.append(team[1,-1].strip(wrestler_start_marker).split(wrestler_end_marker))
        for team in competitors_list:
            if id in team:
                prediction_series['w{wn}m{mn}_allies'.format(wn=wrestler_number, mn=match_number)] = len(team)-1
            else:
                prediction_series['w{wn}m{mn}_opponents'.format(wn=wrestler_number, mn=match_number)] += len(team)
            
        if id in competitors_list[0]:
            prediction_series['w{wn}m{mn}_won_match'.format(wn=wrestler_number, mn=match_number)] = 1
        else:
            prediction_series['w{wn}m{mn}_won_match'.format(wn=wrestler_number, mn=match_number)] = 0
        
    return prediction_series
    
if __name__ == '__main__':
    pass
