
import configparser
import records
import tensorflow as tf
import pandas as pd

config = configparser.ConfigParser()
config.read('config.ini')

team_start_marker = config['all files']['team_start_marker']
team_end_marker = config['all files']['team_end_marker']
wrestler_start_marker = config['all files']['wrestler_start_marker']
wrestler_end_marker = config['all files']['wrestler_end_marker']
db_url = config['all files']['db_url']

def competitor_string_to_list(competitor_string):
    competitor_list = []
    for team in competitor_string.split(team_end_marker)[:-1]:
        competitor_list.append(team.replace(wrestler_start_marker, '').split(wrestler_end_marker)[:-1])
    
    return competitor_list


def competitor_list_to_dict(competitor_list, max_number_of_wrestlers):
    competitor_dict = {}
    for key_number in range(max_number_of_wrestlers):
        competitor_dict[key_number] = 0

    unstacked_list = []
    for team in competitor_list:
        unstacked_list.extend(team)

    for key_number, id in enumerate(unstacked_list):
        competitor_dict[key_number] = id

    return competitor_dict


def clean_text(text):
    import unicodedata
    temp_string = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode()
    temp_string = temp_string.lower()
    temp_string = temp_string.replace("'", "")
    temp_string = temp_string.replace('"', "")
    temp_string = temp_string.replace('.', "")
    return temp_string


def lookup(alias, db=records.Database(db_url=db_url)):
    assert isinstance(alias, str)
    cleaned_alias = clean_text(alias)
    id_dict = db.query("SELECT id FROM alias_table WHERE alias = '{alias}';".format(alias=cleaned_alias), fetchall=True).as_dict()
    if id_dict:
        id = id_dict.get('id')
    else:
        id = 0
        
    return id


def prediction_series_dict(number_of_history_matches, max_number_of_wrestlers):
    wrestler_index_template_list = ['id', 'dob', 'nationality']
    history_match_index_template_list = ['days_since_match', 'won_match', 'wintype', 'title', 'matchtype', 'opponents', 'allies']
    index_dict = {'current_title': 0, 'current_matchtype': 0}

    for wrestler_number in range(max_number_of_wrestlers):
        index_dict['w{wn}_current_allies'.format(wn=wrestler_number)] = 0
        index_dict['w{wn}_current_opponents'.format(wn=wrestler_number)] = 0

    for index_wrestler_number in range(max_number_of_wrestlers):
        for wrestler_template in wrestler_index_template_list:
            index_dict["w{wn}_{temp}".format(wn=index_wrestler_number, temp=wrestler_template)] = 0
            for history_number in range(number_of_history_matches):
                for history_template in history_match_index_template_list:
                    index_dict["w{wn}m{mn}_{temp}".format(wn=index_wrestler_number, mn=history_number, temp=history_template)] = 0

    return index_dict


def make_prediction_series(id, series_type, index_dict, db=records.Database(db_url=db_url), event_date=29991231, number_of_history_matches=10, max_number_of_wrestlers=75):
    assert series_type in ['test', 'train', 'predict', 'validate']
    matches_query_string = """
    SELECT m.date, m.wintype, m.titles, m.matchtype, t.competitors
    FROM match_table m JOIN team_table t ON m.match_id = t.match_id
    WHERE date < {event} AND m.match_id IN (
        SELECT match_id FROM team_table WHERE competitors LIKE '%{start}{id}{end}%'
    )
    ORDER BY date DESC
    LIMIT {limit}"""

    if series_type in ['test', 'train', 'validate']:
        matches_query = db.query(matches_query_string.format(
            event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches+1
        )).as_dict()
        current_match = matches_query.pop()
        index_dict['current_wintype'] = current_match['wintype']
        index_dict['current_titles'] = current_match['titles']
        index_dict['current_matchtype'] = current_match['matchtype']

        current_match_competitors_list = competitor_string_to_list(current_match['competitors'])
        current_match_competitor_dict = competitor_list_to_dict(current_match_competitors_list, max_number_of_wrestlers=max_number_of_wrestlers)
        for wrestler_number in range(max_number_of_wrestlers):
            if current_match_competitor_dict[wrestler_number] == 0:
                break
            for team in current_match_competitors_list:
                if current_match_competitor_dict[wrestler_number] in team:
                    index_dict['w{wn}_current_allies'.format(wn=wrestler_number)] = len(team) - 1
                else:
                    index_dict['w{wn}_current_opponents'.format(wn=wrestler_number)] += len(team)

    else:
        matches_query = db.query(matches_query_string.format(
            event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches
        )).as_dict()

    for match_number, match in enumerate(matches_query):
        competitor_list = competitor_string_to_list(match['competitors'])
        competitor_dict = competitor_list_to_dict(competitor_list, max_number_of_wrestlers=max_number_of_wrestlers)
        for wrestler_number in range(max_number_of_wrestlers):
            if competitor_dict[wrestler_number] == 0:
                break
            else:
                index_dict['w{wn}m{mn}_id'.format(wn=wrestler_number, mn=match_number)] = competitor_dict[wrestler_number]
                stats_query = db.query("SELECT dob, nationality FROM wrestler_table WHERE id = '{id}'".format(id=competitor_dict[wrestler_number])).as_dict()
                for stats in stats_query:
                    index_dict['w{wn}m{mn}_dob'.format(wn=wrestler_number, mn=match_number)] = stats.get('dob', 0)
                    index_dict['w{wn}m{mn}_nationality'.format(wn=wrestler_number, mn=match_number)] = stats.get('nationality', 0)
                    index_dict['w{wn}m{mn}_days_since_match'.format(wn=wrestler_number, mn=match_number)] = event_date - match['date']
                    index_dict['w{wn}m{mn}_wintype'.format(wn=wrestler_number, mn=match_number)] = match['wintype']
                    index_dict['w{wn}m{mn}_title'.format(wn=wrestler_number, mn=match_number)] = match['titles']
                    index_dict['w{wn}m{mn}_matchtype'.format(wn=wrestler_number, mn=match_number)] = match['matchtype']

                for team in competitor_list:
                    if competitor_dict[wrestler_number] in team:
                        index_dict['w{wn}m{mn}_allies'.format(wn=wrestler_number, mn=match_number)] = len(team) - 1
                    else:
                        index_dict['w{wn}m{mn}_opponents'.format(wn=wrestler_number, mn=match_number)] += len(team)

                if id in competitor_list[0]:
                    index_dict['w{wn}m{mn}_won_match'.format(wn=wrestler_number, mn=match_number)] = 1
                else:
                    index_dict['w{wn}m{mn}_won_match'.format(wn=wrestler_number, mn=match_number)] = 0
        
    return index_dict


def make_dataset_dict(db=records.Database(db_url=db_url), number_of_matches=1000, number_of_history_matches=10, max_number_of_wrestlers=75):
    dataset_dict = {'test': None, 'train': None, 'validate': None}
    blank_dict = prediction_series_dict(number_of_history_matches=number_of_history_matches, max_number_of_wrestlers=max_number_of_wrestlers)

    for key in dataset_dict.keys():
        match_query = db.query("SELECT m.date AS date, t.competitors AS competitors FROM match_table m JOIN team_table t ON t.match_id = m.match_id WHERE id IN (SELECT id FROM match_table ORDER BY RANDOM() LIMIT {limit})".format(limit=number_of_matches)).as_dict()
        dict_list = []
        for match in match_query:
            teams_list = competitor_string_to_list(match['competitors'])
            for team in teams_list:
                for id in team:
                    temp_dict = make_prediction_series(id=id, series_type=key, index_dict=blank_dict.copy(), db=db, event_date=match['date'] + 1, number_of_history_matches=number_of_history_matches, max_number_of_wrestlers=max_number_of_wrestlers)
                    dict_list.append(temp_dict)
        temp_dataset = pd.DataFrame.from_dict(dict_list)
        dataset_dict[key] = temp_dataset
        
    return dataset_dict
    

if __name__ == '__main__':
    db = records.Database(db_url=db_url)
    dataset_dict = make_dataset_dict()
    pass
