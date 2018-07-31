
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
    
def make_prediction_series(alias, event_date, db=29991231, number_of_history_matches=10):
    from math import log10, ceil
    id = lookup(alias, db)
    
    index_list = ['wrestler_id', 'dob', 'nationality']
    index_template_list = ['days_since_match', 'won_match', 'wintype_match', 'title_match', 'matchtype_match', 'opponents_match', 'allies_match']
    padding = int(ceil(log10(number_of_history_matches)))
    for number in range(number_of_history_matches):
        padded_number = str(number).zfill(padding)
        for template in index_template_list:
            index_list.extend("".join([template, padded_number]))

    prediction_series = pd.Series(0, index=index_list)
    stats_query = db.query("SELECT dob, nationality FROM wrestlers_table WHERE id = '{id}'".format(id=id)).as_dict()
    
    matches_query_string = """SELECT m.date, m.wintype, m.title, m.matchtype, c.competitors FROM match_table m JOIN competitor_table c ON m.id = c.id WHERE date < {event} AND id IN (
    SELECT id as match_id FROM competitors_table WHERE competitors LIKE '%{start}{id}{end}%'
    )
    ORDER BY date DESC, LIMIT {limit}".format(start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches)""".format(
        event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_markers
    )
    matches_query = db.query(matches_query_string).as_dict()
    
    prediction_series['id'] = id
    prediction_series['dob'] = stats_query.get('dob',0)
    prediction_series['nationality'] = stats_query.get('nationality',0)
    for number, match in enumerate(matches_query):
        padded_number = str(number).zfill(padding)

        prediction_series['days_since_match'.format(padded_number)] = event_date - match['date']
        prediction_series['wintype_match'.format(padded_number)] = match['wintype']
        prediction_series['title_match'.format(padded_number)] = match['title']
        prediction_series['matchtype_match'.format(padded_number)] = match['matchtype']
        
        competitors_list = []
        for team in match['competitors'].split(team_end_marker):
            competitors_list.append(team[1,-1].strip(wrestler_start_marker).split(wrestler_end_marker))
        for team in competitors_list:
            if id in team:
                prediction_series['allies_match'.format(padded_number)] = len(team)-1
            else:
                prediction_series['opponents_match'.format(padded_number)] += len(team)
            
        if id in competitors_list[0]:
            prediction_series['won_match'.format(padded_number)] = 1
        else:
            prediction_series['won_match'.format(padded_number)] = 0
        
    return prediction_series
    
if __name__ == '__main__':
    pass
