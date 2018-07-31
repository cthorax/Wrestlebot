
import os
import random
import records
import tensorflow as tf
import pandas as pd
import numpy as np


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
    
    global max_team_size
    global max_competitors
    wrestler_query = "SELECT id FROM teams_table WHERE wrestler00 = '{id}'".format(id=id)
    for wrestler_number in range(1, max_team_size):
        wrestler_query_template = "wrestler{number} = '{id}'"
        wrestler_query = " OR ".join([wrestler_query, wrestler_query_template.format(id=id, number=str(wrestler_number).zfill(2))])
                                      
    teams_query = db.query(wrestler_query).as_dict()
    competitors_query = db.query("SELECT id FROM competitors_table WHERE teamXX = '{id}'".format(id=id)).as_dict()
    history_query = db.query("SELECT date, wintype, title, matchtype, competitors WHERE date < {event_date} AND competitors IN (\nSELECT id FROM competitors_table WHERE  ) FROM match_table ORDER BY date DESC".format()).as_dict()

    prediction_series['id'] = id
    prediction_series['dob'] = stats_query.get('dob',0)
    prediction_series['nationality'] = stats_query.get('nationality',0)
                              
                              
    
if __name__ == '__main__':
    pass
