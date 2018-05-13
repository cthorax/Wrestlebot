import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pymongo


mongo_address = os.getenv("mongo_address")
client = pymongo.MongoClient(mongo_address)
client.is_mongos  # blocks until connected to server
db = client["Wrastlin"]
match_collection = db["Matches 3"]
wrestler_collection = db["Wrestlers"]

wintype_antidict = {
    51: 'loss',
    52: 'loss via pinfall',
    53: 'loss via submission',
    54: 'loss via disqualification',
    55: 'loss via forfeit',
    56: 'loss count-out',
    57: 'loss via ko',
    58: 'loss via tko',
    1: 'win',
    2: 'win via pinfall',
    3: 'win via submission',
    4: 'win via disqualification',
    5: 'win via forfeit',
    6: 'win via count-out',
    7: 'win via ko',
    8: 'win via tko',
    90: 'draw',
    91: 'draw, no contest',
    92: 'draw, double count-out',
    93: 'draw, time out',
    94: "draw, double disqualification",
    95: "draw, double pin",
    99: "none, no match"
}
title_antidict = {
    0: 'none',
    1: 'wwe universal title (raw)',
    2: 'wwe intercontinental title (raw)',
    3: 'wwe raw womens title',
    4: 'wwe raw tag team title',

    11: 'wwe championship title (smackdown)',
    12: 'wwe united states title (smackdown)',
    13: 'wwe smackdown woments title',
    14: 'wwe smackdown tag team title',

    20: 'both smackdown titles',

    31: 'nxt title',
    32: 'nxt north america title',
    33: 'wwe uk title',
    34: 'nxt womens title',
    35: 'nxt tag team title',

    99: 'unknown title'
}

blank_match = {
    'teams': [],
    'wintype': None,
    'matchtype': None,
    'wrestlers': [],
    'winners': [],
    'duration': 0,
    'date': None,
    'titles': 'none',
    '_id': None
}


def db_call(db=None, query=None, mode='find', retries=None, verbose=False):
    assert db
    assert mode in ['find', 'aggregate']
    assert isinstance(retries, int) or retries is None

    result = None
    failures = 0

    while not result:
        try:
            if mode == 'find':
                if query:
                    result = db.find(query)
                else:
                    result = db.find()
            elif mode == 'aggregate':
                if query:
                    result = db.aggregate(query)
                else:
                    result = db.aggregate()

        except (pymongo.errors.AutoReconnect, TimeoutError):
            failures += 1
            if retries:
                if failures > retries:
                    raise pymongo.errors.AutoReconnect
                else:
                    if verbose:
                        print ('db call timed out. retrying.')
                        result = None

    return result


def history_dataset():
    columns = [
        '0_wintype', '0_matchtype', '0_duration', '0_titles', '0_number_of_teams', '0_opponents', '0_allies',
        '1_wintype', '1_matchtype', '1_duration', '1_titles', '1_number_of_teams', '1_opponents', '1_allies',
        '2_wintype', '2_matchtype', '2_duration', '2_titles', '2_number_of_teams', '2_opponents', '2_allies',
        '3_wintype', '3_matchtype', '3_duration', '3_titles', '3_number_of_teams', '3_opponents', '3_allies',
        '4_wintype', '4_matchtype', '4_duration', '4_titles', '4_number_of_teams', '4_opponents', '4_allies',
        '5_wintype', '5_matchtype', '5_duration', '5_titles', '5_number_of_teams', '5_opponents', '5_allies',
        '6_wintype', '6_matchtype', '6_duration', '6_titles', '6_number_of_teams', '6_opponents', '6_allies',
        '7_wintype', '7_matchtype', '7_duration', '7_titles', '7_number_of_teams', '7_opponents', '7_allies',
        '8_wintype', '8_matchtype', '8_duration', '8_titles', '8_number_of_teams', '8_opponents', '8_allies',
        '9_wintype', '9_matchtype', '9_duration', '9_titles', '9_number_of_teams', '9_opponents', '9_allies',
        'match_id', 'matchtype', 'titles', 'number_of_teams', 'opponents', 'allies', 'wintype']
    return pd.DataFrame(columns=columns)


class Wrestler(object):
    def __init__(self, name, match_dict=None):
        self.source_match = match_dict
        self.is_training = match_dict.get('is_training', False)
        self.circa = match_dict.get('date', 30001231)
        self.name = name.lower()
        self.history = history_dataset()
        self.update()

    def update(self):
        # this will update all relevant wrestler stats when called

        try:
            aliases = list(db_call(db=wrestler_collection, query={"name": self.name}, mode='find', verbose=verbose))
            assert aliases  # checks the DB for an instance of the wrestler, and fails if none exists
            self.aliases = aliases[0]['name']
            self.get_history()

        except AssertionError:
            # deletion doesn't work since this is done in a reference to itself, so clearing all values and removing Nones later will have to suffice
            if verbose:
                print("wrestler '{}' does not exist.".format(self.name))
            self.source_match = None
            self.is_training = None
            self.circa = None
            self.name = ''
            self.history = None


    def get_history(self):
        # changes list from MongoDB results into Dataframe from pandas

        history = []
        for alias in self.aliases:
            if self.is_training:
                query = {'$and':
                    [
                        {"_id": {'$ne': self.source_match['_id']}},
                        {"wrestlers": alias},
                        {"date": {'$lt': self.circa}}
                    ]
                }
            else:
                query = {'$and':
                    [
                        {"wrestlers": alias},
                        {"date": {'$lt': self.circa}}
                    ]
                }

            temp_history = list(db_call(db=match_collection, query=query, mode='find', verbose=verbose).limit(10).sort("date", pymongo.DESCENDING))
            history.extend(temp_history)
        history = sorted(history, key=lambda x: x['date'], reverse=True)[:10]
        # sorts the list by date descending and then truncates to number_of_matches entries
        history = sorted(history, key=lambda x: x['date'], reverse=False)
        # sorts it back to date ascending. since the optional predictor values will be at the front,
        # we need to work in reverse as we build.

        while len(history) < 10:
            history.append(blank_match)
        # ensures dataframes are all the same size

        source_dict = {}
        if self.is_training:    # source_match is an actual match from the db
            for alias in self.aliases:
                if alias in self.source_match['wrestlers']:
                    current_alias = alias
                    break

            is_winner = current_alias in self.source_match['winners']
            wintype = self.source_match.get('wintype', "none, no match")
            adjusted_value = wintype_dict.get(wintype, 99)
            if adjusted_value >= 90 or is_winner:
                pass
            else:
                adjusted_value += 50
            source_dict['wintype'] = [adjusted_value]

        else:   # source_match is the source_dict from instantiation of match. no match_id, no wintype.
            current_alias = self.name

        teams = self.source_match.get('teams', [])
        number_of_teams = len(teams)
        opponents = 0
        allies = 0

        for team in teams:
            if isinstance(team, dict):
                if current_alias in team['members']:
                    allies = len(team) - 1
                else:
                    opponents += len(team)
            elif isinstance(team, list):
                if current_alias in team:
                    allies = len(team) - 1
                else:
                    opponents += len(team)
            elif isinstance(team, str):
                if current_alias == team:
                    allies = 0
                else:
                    opponents += 1

        source_dict['number_of_teams'] = [number_of_teams]
        source_dict['opponents'] = [opponents]
        source_dict['allies'] = [allies]

        titles = title_dict.get(self.source_match.get('titles', 'none').split('\n(title change)')[0], 0)
        source_dict['titles'] = [titles]

        duration = self.source_match.get('duration', 0)
        source_dict['duration'] = [duration]

        matchtype = self.source_match.get('matchtype', 'normal')
        source_dict['matchtype'] = [hash(matchtype)]

        source_dataframe = pd.DataFrame.from_dict(source_dict)
        match_id = [self.source_match.get('_id', 'no id')]

        matches_dict = {}
        for match_number, match in enumerate(history):
            for alias in self.aliases:
                wrestler_list = match.get('wrestlers', [])
                if alias in wrestler_list:
                    current_alias = alias
                    break

            is_winner = current_alias in match['winners']

            teams = match.get('teams', [])
            number_of_teams = len(teams)
            opponents = 0
            allies = 0
            for team in teams:
                if current_alias in team:
                    allies = len(team) - 1
                else:
                    opponents += len(team)

            matches_dict["".join([str(match_number), "_number_of_teams"])] = [number_of_teams]
            matches_dict["".join([str(match_number), "_opponents"])] = [opponents]
            matches_dict["".join([str(match_number), "_allies"])] = [allies]

            titles = match.get('titles', '').split('\n(title change)')[0]
            matches_dict["".join([str(match_number), "_titles"])] = [title_dict.get(titles, 0)]

            matchtype = match.get('matchtype', 'normal')
            matches_dict["".join([str(match_number), "_matchtype"])] = [hash(matchtype)]

            duration = match.get('duration', 0)
            matches_dict["".join([str(match_number), "_duration"])] = [duration]

            wintype = match.get('wintype', None)
            adjusted_value = wintype_dict.get(wintype, 99)
            if adjusted_value >= 90 or is_winner:
                pass
            else:
                adjusted_value += 50

            matches_dict["".join([str(match_number), "_wintype"])] = [adjusted_value]

        matches_dataframe = pd.DataFrame.from_dict(matches_dict)
        history_dataframe = matches_dataframe.join(source_dataframe).astype(int)
        history_dataframe = history_dataframe.assign(match_id=match_id)

        self.history = history_dataframe


class Team(object):
    def __init__(self, team_dict=None, match_dict=None):
        self.team_dict = {
            'teams': {
                'name': 'empty team',
                'members': []
            }
        }
        self.match_dict = match_dict

        if team_dict:
            self.team_dict['teams']['name'] = team_dict.get('name', 'empty team')
            self.team_dict['teams']['members'] = team_dict.get('teams', {}).get('members', [])
            self.update()

    def add_members(self, new_members):
        if isinstance(new_members, list):
            self.team_dict['teams']['members'].extend(new_members)
        elif isinstance(new_members, Wrestler):
            if new_members.name:
                self.team_dict['teams']['members'].append(new_members)
        self.update()

    def update(self):
        # this will update all relevant team stats when called.
        for index, member in enumerate(self.team_dict['teams']['members']):
            if isinstance(member, Wrestler):
                pass
            elif isinstance(member, str):
                temp_member = Wrestler(name=member, match_dict=self.match_dict)
                self.team_dict['teams']['members'][index] = temp_member

        if self.team_dict['teams']['name'] == 'empty team' or not self.team_dict['teams']['name']:
            team_name = ""

            for wrestler in self.team_dict['teams']['members']:
                team_name = ", ".join([team_name, wrestler.name])

            team_name = team_name[2:]  # removes extra ', ' from front
            team_name = ', & '.join(team_name.rsplit(', ',1))
            if team_name.count(',') == 1:  # oxford comma fixer
                team_name = team_name.replace(',', '')

            self.team_dict['teams']['name'] = team_name


class Match(object):
    def __init__(self, match_dict=None):
        if match_dict:
            self.match_dict = {}
            self.match_dict['name'] = match_dict.get('name', 'empty match')
            self.match_dict['titles'] = match_dict.get('titles', 'unknown title')
            self.match_dict['matchtype'] = match_dict.get('matchtype', 'normal')
            self.match_dict['teams'] = match_dict.get('teams', [])

            self.match_dict = match_dict
            self.update()
        else:
            self.match_dict = {
                'name': 'empty match',
                'titles': 'none',
                'matchtype': 'normal',
                'teams': []
            }

    def add_teams(self, new_teams):
        if isinstance(new_teams, list):
            self.match_dict['teams'].extend(new_teams)
        else:
            self.match_dict['teams'].append(new_teams)

    def update(self):
        #this will update all relevant Match stats when called.
        if self.match_dict['teams']:
            for index, team in enumerate(self.match_dict['teams']):
                if isinstance(team, Team):
                    pass
                elif isinstance(team, dict):
                    temp_dict = {
                        'teams': team
                    }

                    temp_team = Team(team_dict=temp_dict, match_dict=self.match_dict)
                    self.match_dict['teams'][index] = temp_team

        if self.match_dict.get('name', 'empty match') == 'empty match':
            match_name = ""

            for team in self.match_dict['teams']:
                match_name = " VS ".join([match_name, team.team_dict['teams']['name']])

            self.match_dict['name'] = match_name[4:]  # removes extra ' VS ' from front

    def base_predict(self, model):
        assert isinstance(model, Model)

        self.predictions = None
        # create the team predictions from the wrestler predictions
        for team in self.match_dict['teams']:
            team.predictions = []
            team.naive_win = []
            team.naive_lose = []
            team.naive_draw = []

            for wrestler in team.team_dict.get('teams', []).get('members', []):
                dataset = wrestler.history.copy()

                if 'match_id' in dataset.keys():
                    dataset.pop('match_id')
                if 'wintype' in dataset.keys():
                    dataset.pop('wintype')

                wrestler.predictions = model.make_prediction(dataset)
                wrestler.win = np.sum(wrestler.predictions[1:8])*100
                wrestler.lose = np.sum(wrestler.predictions[51:58])*100
                wrestler.draw = np.sum(wrestler.predictions[91:95])*100

                team.predictions.append(wrestler.predictions)
                team.naive_win.append(wrestler.win)
                team.naive_lose.append(wrestler.lose)
                team.naive_draw.append(wrestler.draw)

            team.predictions = np.average(team.predictions, axis=0)
            team.naive_win = np.average(team.naive_win)
            team.naive_lose = np.average(team.naive_lose)
            team.naive_draw = np.average(team.naive_draw)

        win_product = []
        lose_product = []
        draw_product = []
        for team in self.match_dict['teams']:
            win_product.append(team.naive_win)
            lose_product.append(team.naive_lose)
            draw_product.append(team.naive_draw)

        win_product = np.prod(win_product)
        lose_product = np.prod(lose_product)
        draw_product = np.prod(draw_product)

        self.predicted_winner = [None, 0]
        self.predicted_wintype = None
        norm = 0
        for team in self.match_dict['teams']:
            team.true_win = np.true_divide(lose_product, team.naive_lose)
            team.true_win = np.prod([team.true_win, team.naive_win])

            team.true_lose = np.true_divide(win_product, team.naive_win)
            team.true_lose = np.prod([team.true_lose, team.naive_lose])

            team.true_draw = draw_product
            norm = np.sum([team.true_win, norm])

        norm = np.sum([team.true_draw, norm])     # true_draw should be the same for all teams, so this only needs to be added once.

        for team in self.match_dict['teams']:
            if self.predicted_winner[1] <= team.true_win:
                self.predicted_winner = [team.team_dict.get('teams', []).get('name', None), team.true_win]

                for wintype, type_probability in enumerate(team.predictions[1:9], start=1):
                    if type_probability == max(team.predictions[1:9]):
                        self.predicted_wintype = wintype
                        break

        prediction_dict = {
            'prediction_string': "using {model_name} model:\npredicted winner of '{match_name}' is\n{winner_name} ({winner_percent:.3%}).\nMost likely wintype is {wintype}.\n",
            'model_name': model.name,
            'match_name': self.match_dict['name'],
            'winner_name': self.predicted_winner[0],
            'winner_percent': self.predicted_winner[1] / norm,
            'wintype': wintype_antidict[self.predicted_wintype]
        }

        return prediction_dict

    def aggregate_predict(self, model_list):
        for model in model_list:
            assert isinstance(model, Model) or model is None

        winner = compare_models(model_list=model_list, verbose=verbose)
        prediction_list = []

        team_dict = {}
        for team in self.match_dict['teams']:
            team_name = team.team_dict['teams']['name']
            team_dict[team_name] = 0

        for model in model_list:
            if model:
                temp_prediction = self.base_predict(model=model)
                if model is winner:
                    temp_prediction['is_winner'] = True
                prediction_list.append(temp_prediction)

        final_prediction = None
        for prediction in prediction_list:
            team_name = prediction['winner_name']
            team_dict[team_name] += 1 + prediction.get('is_winner', 0)
            current_count = team_dict[team_name]

            if final_prediction:
                if current_count > final_count:
                    final_prediction = prediction
                    final_count = team_dict[final_prediction['winner_name']]
                else:
                    if current_count == final_count and prediction.get('is_winner', None):
                        final_prediction = prediction
                        final_count = team_dict[final_prediction['winner_name']]

            else:
                final_prediction = prediction
                final_count = team_dict[final_prediction['winner_name']]

        return final_prediction


class Event(object):
    def __init__(self, event_dict=None):
        if event_dict:
            self.event_dict = {}
            self.event_dict['name'] = event_dict.get('name', 'empty event')
            self.event_dict['date'] = event_dict.get('date', 30010101)
            self.event_dict['matches'] = event_dict.get('matches', [])
            self.update()
        else:
            self.event_dict = {
                'name': 'empty event',
                'date': 'none',
                'matches': []
            }

    def add_event(self, new_matches):
        if isinstance(new_matches, list):
            self.event_dict['matches'].extend(new_matches)
        else:
            self.event_dict['matches'].append(new_matches)

    def update(self):
        #this will update all relevant Match stats when called.
        if self.event_dict['matches']:
            for index, match in enumerate(self.event_dict['matches']):
                if isinstance(match, Match):
                    pass
                elif isinstance(match, dict):
                    temp_dict = match
                    temp_dict['date'] = self.event_dict['date']

                    temp_match = Match(match_dict=temp_dict)
                    self.event_dict['matches'][index] = temp_match


class Dataset(object):
    def __init__(self, limit=None, query=None, backup_every=50, name=None, verbose=False):
        assert isinstance(limit, int) or not limit

        self.name = name
        self.backup_every = backup_every

        self.train_dataset = history_dataset()
        self.train_backup_file = 'G:\wrestlebot\csv query backups\{} - train.csv'.format('{}'.format(query).replace(":", '').replace("'", ''))
        self.train_offset = 0

        self.test_dataset = history_dataset()
        self.test_backup_file = 'G:\wrestlebot\csv query backups\{} - test.csv'.format('{}'.format(query).replace(":", '').replace("'", ''))
        self.test_offset = 0

        self.validate_dataset = history_dataset()
        self.validate_backup_file = 'G:\wrestlebot\csv query backups\{} - validate.csv'.format('{}'.format(query).replace(":", '').replace("'", ''))
        self.validate_offset = 0

        self.match_list = []

        for dataset_name in ['train', 'test', 'validate']:
            # self.csv_input(dataset_name)
            self.csv_input(dataset_name)

        results, size, oversize = self.get_results(limit=limit, query=query)

        if results:     # all this can be skipped if there's no elements in results, because we then know we're already loaded up.
            self.parse_results(results=results, size=size, oversize=oversize)

        for dataset in [self.train_dataset, self.test_dataset, self.validate_dataset]:
            if 'match_id' in dataset.keys():
                dataset.pop('match_id')

    def get_results(self, limit, query):
        if limit:
            size = limit // 3
            # if the backup files for some but not all of the sets are over the limit, don't truncate to save loading.
            train_has = min(self.train_offset, size)
            test_has = min(self.test_offset, size)
            validate_has = min(self.validate_offset, size)

            true_limit = limit - train_has - test_has - validate_has
            if true_limit > 0:
                if query:
                    metaquery = [
                        {'$match': query},
                        {'$sample': {'size': true_limit}}
                    ]
                    results = list(db_call(db=match_collection, query=metaquery, mode='aggregate', verbose=verbose))

                else:
                    metaquery = [
                        {'$sample': {'size': true_limit}}
                    ]
                    results = list(db_call(db=match_collection, query=metaquery, mode='aggregate', verbose=verbose))

            else:
                results = []
                if verbose:
                    print("datasets restored from file.".format())
        else:
            if query:
                results = list(db_call(db=match_collection, query=query, mode='find', verbose=verbose))
            else:
                results = list(db_call(db=match_collection, query=None, mode='find', verbose=verbose))
            size = len(results) // 3

        if len(results) >= true_limit:
            oversize = True
        else:
            oversize = False
        return results, size, oversize

    def parse_results(self, results, size, oversize):
        random.shuffle(results)
        train_results = []
        test_results = []
        validate_results = []

        train_needs = max(size - self.train_offset, 0)
        test_needs = max(size - self.test_offset, 0)
        validate_needs = max(size - self.validate_offset, 0)

        if oversize:
            while len(train_results) < train_needs:
                train_results.append(results.pop())
            while len(test_results) < test_needs:
                test_results.append(results.pop())
            while len(validate_results) < validate_needs:
                validate_results.append(results.pop())
        else:
            train_has = min(self.train_offset, size)
            test_has = min(self.test_offset, size)
            validate_has = min(self.validate_offset, size)

            while test_has + train_has + validate_has < 3*size:
                try:
                    if train_has < size and train_has == min(train_has, test_has, validate_has):
                        train_results.append(results.pop())
                        train_has += 1

                    if test_has < size and test_has == min(train_has, test_has, validate_has):
                        test_results.append(results.pop())
                        test_has += 1

                    if validate_has < size and validate_has == min(train_has, test_has, validate_has):
                        validate_results.append(results.pop())
                        validate_has += 1

                except LookupError:
                    break    # when results == [], we're done anyhow

        for dataset_name in ['train', 'test', 'validate']:
            if dataset_name == 'train':
                backup_file = self.train_backup_file
                offset = self.train_offset
                results = train_results
            elif dataset_name == 'test':
                backup_file = self.test_backup_file
                offset = self.test_offset
                results = test_results
            elif dataset_name == 'validate':
                backup_file = self.validate_backup_file
                offset = self.validate_offset
                results = validate_results

            for index, match in enumerate(results):
                match['is_training'] = True
                if verbose:
                    print('\rstarting \'{}\' dataset: item {} of {}'.format(backup_file[:-4], index+1+offset, len(results)+offset), end='')
                if match['_id'] in self.match_list:
                    continue
                else:
                    for name in match['wrestlers']:
                        temp_wrestler = Wrestler(name=name, match_dict=match)
                        self.update(dataset=temp_wrestler.history, mode=dataset_name)
                    if index % self.backup_every == 0 and index > 0:
                        self.csv_output(mode=dataset_name, backup_file=backup_file)
            # final backup after finishing
            self.csv_output(mode=dataset_name, backup_file=backup_file, verbose=verbose)
            if verbose:
                print('\r\'{}\' dataset finished.'.format(backup_file[:-4]))

    def csv_input(self, dataset_name):
        assert dataset_name in ['train', 'test', 'validate']
        if dataset_name == 'train':
            backup_file = self.train_backup_file
        elif dataset_name == 'test':
            backup_file = self.test_backup_file
        elif dataset_name == 'validate':
            backup_file = self.validate_backup_file

        try:
            with open(backup_file, 'r') as dataset_backup:
                dataset = pd.read_csv(dataset_backup)
                self.update(dataset=dataset, mode=dataset_name)

            if dataset_name == 'train':
                match_list = set(self.train_dataset['match_id'])
                self.match_list.extend(match_list)
                self.train_offset = len(match_list)
            elif dataset_name == 'test':
                match_list = set(self.test_dataset['match_id'])
                self.match_list.extend(match_list)
                self.test_offset = len(match_list)
            elif dataset_name == 'validate':
                match_list = set(self.validate_dataset['match_id'])
                self.match_list.extend(match_list)
                self.validate_offset = len(match_list)

        except FileNotFoundError:
            if verbose:
                print("backup file '{}' not found, starting from scratch.".format(backup_file))
            if dataset_name == 'train':
                self.train_offset = 0
            elif dataset_name == 'test':
                self.test_offset = 0
            elif dataset_name == 'validate':
                self.validate_offset = 0

        except NotImplementedError as e:  # not sure what error yet
            if verbose:
                print('file is corrupted. removing.\n', e)
            os.remove(backup_file)
            if dataset_name == 'train':
                self.train_offset = 0
            elif dataset_name == 'test':
                self.test_offset = 0
            elif dataset_name == 'validate':
                self.validate_offset = 0

    def update(self, dataset, mode=None):
        assert mode in ['train', 'test', 'validate']

        if mode == 'train':
            self.train_dataset = self.train_dataset.append(dataset, ignore_index=True)
        elif mode == 'test':
            self.test_dataset = self.test_dataset.append(dataset, ignore_index=True)
        elif mode == 'validate':
            self.validate_dataset = self.validate_dataset.append(dataset, ignore_index=True)

    def csv_output(self, mode=None, backup_file=None, verbose=False):
        assert isinstance(backup_file, str)
        assert mode in ['train', 'test', 'validate']

        if mode == 'train':
            backup_file = self.train_backup_file
            dataset = self.train_dataset

        elif mode == 'test':
            backup_file = self.test_backup_file
            dataset = self.test_dataset

        elif mode == 'validate':
            backup_file = self.validate_backup_file
            dataset = self.validate_dataset

        if not dataset.empty:
            with open(backup_file, 'w') as dataset_backup:
                dataset.to_csv(dataset_backup, index=False, header=True)
                if verbose:
                    print(' - backup written', end='')


class Model(object):
    def __init__(self, batch_size=100, train_steps=1000, model_type='linear', dataset=None, layer_specs=[10, 10, 10], name=None):
        # called when created. instantiated with a model type as string, and a pair of datasets in a dict, each labeled 'test' or 'train'.
        assert isinstance(layer_specs, list)
        assert isinstance(dataset, Dataset)

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.model_type = model_type
        self.test_dataset = dataset.test_dataset
        self.train_dataset = dataset.train_dataset
        self.validate_dataset = dataset.validate_dataset
        self.layer_specs = layer_specs
        self.name = name
        self.save_hash = hash('{} - {}'.format(self.name, self.layer_specs))

        # get datasets to train and test
        (self.train_x, self.train_y), (self.test_x, self.test_y), (self.validate_x, self.validate_y) = self.load_data()

        # train model
        # self.train_model()
        self.train_model(verbose=verbose)

        # evaluate model
        self.assess_model(verbose=verbose)

    def load_data(self, y_name="wintype"):     # when no longer testing, change limit probably
        # right now this only works for numeric values
        train_x = self.train_dataset.astype(int)
        train_y = self.train_dataset.get(y_name).astype(int)

        test_x = self.test_dataset.astype(int)
        test_y = self.test_dataset.get(y_name).astype(int)

        validate_x = self.validate_dataset.astype(int)
        validate_y = self.validate_dataset.get(y_name).astype(int)

        return (train_x, train_y), (test_x, test_y), (validate_x, validate_y)

    def train_input_fn(self, features, labels):
        #stolen from the iris_data tutorial
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)

        # Return the dataset.
        return dataset

    def eval_input_fn(self, features, labels):
        #stolen from the iris_data tutorial
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert self.batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(self.batch_size)

        # Return the dataset.
        return dataset

    def load_model(self, load_file):
        # loads a saved model
        pass

    def save_model(self):
        # saves a model
        pass

    def train_model(self, verbose=False):
        # calls input_to_model to train a new model

        # define feature columns
        numeric_feature_columns = []
        wide_categorical_columns = []
        deep_categorical_columns = []

        numeric_columns = [
            '0_duration', '0_number_of_teams', '0_opponents', '0_allies',
            '1_duration', '1_number_of_teams', '1_opponents', '1_allies',
            '2_duration', '2_number_of_teams', '2_opponents', '2_allies',
            '3_duration', '3_number_of_teams', '3_opponents', '3_allies',
            '4_duration', '4_number_of_teams', '4_opponents', '4_allies',
            '5_duration', '5_number_of_teams', '5_opponents', '5_allies',
            '6_duration', '6_number_of_teams', '6_opponents', '6_allies',
            '7_duration', '7_number_of_teams', '7_opponents', '7_allies',
            '8_duration', '8_number_of_teams', '8_opponents', '8_allies',
            '9_duration', '9_number_of_teams', '9_opponents', '9_allies',
            'number_of_teams', 'opponents', 'allies'
        ]
        for key in numeric_columns:
            numeric_feature_columns.append(
                tf.feature_column.numeric_column(key=key)
            )

        win_columns = [
            '0_wintype', '1_wintype', '2_wintype', '3_wintype', '4_wintype',
            '5_wintype', '6_wintype', '7_wintype', '8_wintype', '9_wintype',
        ]
        title_columns = [
            '0_titles', '1_titles', '2_titles', '3_titles', '4_titles',
            '5_titles', '6_titles', '7_titles', '8_titles', '9_titles',
            'titles'
        ]
        for columns, vocab in [(win_columns, wintype_antidict), (title_columns, title_antidict)]:
            size = max(vocab.keys())+1      # must be smaller than number of buckets (i guess bucket zero isn't counted?)
            for key in columns:
                wide_categorical_columns.append(
                    tf.feature_column.categorical_column_with_identity(
                        key=key,
                        num_buckets=size
                    )
                )

        hashed_columns = [
            '0_matchtype', '1_matchtype', '2_matchtype', '3_matchtype', '4_matchtype',
            '5_matchtype', '6_matchtype', '7_matchtype', '9_matchtype',
            'matchtype'
        ]

        for key in hashed_columns:
            wide_categorical_columns.append(
                tf.feature_column.categorical_column_with_hash_bucket(
                    key=key,
                    hash_bucket_size=10000,
                    dtype=tf.int32
                )
            )

        for column in wide_categorical_columns:
            deep_categorical_columns.append(
                tf.feature_column.indicator_column(
                    column
                )
            )

        if self.model_type == 'linear':
            classifier = tf.estimator.LinearClassifier(
                model_dir='G:\wrestlebot\linear models\{}'.format(self.save_hash),
                feature_columns=numeric_feature_columns + wide_categorical_columns,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'deep':
            classifier = tf.estimator.DNNClassifier(
                model_dir='G:\wrestlebot\deep models\{}'.format(self.save_hash),
                feature_columns=numeric_feature_columns + deep_categorical_columns,
                hidden_units=self.layer_specs,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'hybrid':
            classifier = tf.estimator.DNNLinearCombinedClassifier(
                model_dir='G:\wrestlebot\hybrid models\{}'.format(self.save_hash),
                linear_feature_columns=numeric_feature_columns + wide_categorical_columns,
                dnn_feature_columns=numeric_feature_columns + deep_categorical_columns,
                dnn_hidden_units=self.layer_specs,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )

        # Train the Model.
        if verbose:
            print('training the {} classifier \'{}\' for {} steps.'.format(self.model_type, self.name, self.train_steps))
        classifier.train(
            input_fn=lambda: self.train_input_fn(
                self.train_x,
                self.train_y
            ),
            steps=self.train_steps
        )

        self.classifier = classifier

    def make_prediction(self, predict_x, labels=None):
        # calls input_to_model to make a prediction
        predictions = self.classifier.predict(
            input_fn=lambda: self.eval_input_fn(
                predict_x,
                labels=labels
            ),
        )

        predictions = predictions.__next__()
        probabilities = predictions['probabilities']

        return probabilities

    def assess_model(self, verbose=False):
        # Evaluate the model.
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.eval_input_fn(
                self.test_x,
                self.test_y
        ))

        accuracy = eval_result['accuracy']
        if verbose:
            print('Test set accuracy for \'{}\' model: {:.3%}\n'.format(self.name, accuracy))
        self.model_accuracy = accuracy

    def compare(self, validate_x, validate_y):
        # Evaluate the model based on the validate set to compare two models.
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.eval_input_fn(
                validate_x,
                validate_y
        ))

        return eval_result['accuracy']


def compare_models(model_list, verbose=False):
    assert isinstance(model_list, list)
    for model in model_list:
        assert isinstance(model, Model) or model is None

    best = [0, 0]
    winner = None

    for model in model_list:
        if model is None:
            continue
        else:
            try:
                subset_result = model.compare(subset_x, subset_y)
            except UnboundLocalError:       # only will occur if the first model in the set. Since this should be the smallest subset, this behavior saves processing.
                subset_x = model.validate_x
                subset_y = model.validate_y
                subset_result = model.compare(subset_x, subset_y)
                model.validate_accuracy = subset_result

            try:
                assert model.validate_accuracy
            except AttributeError:
                model.validate_accuracy = model.compare(model.validate_x, model.validate_y)

            if verbose:
                print('\'{}\' model:\nown set: {:.3%}\nsubset: {:.3%}\n'.format(model.name, model.validate_accuracy, subset_result))
            if subset_result > best[0]:
                best = [subset_result, model.validate_accuracy]
                winner = model
            elif subset_result == best[0] and model.validate_accuracy > best[1]:
                best = [subset_result, model.validate_accuracy]
                winner = model

    if verbose:
        print('most accurate model is \'{}\'.\n'.format(winner.name))
    return winner


def generate_queries(match):
    assert isinstance(match, Match)
    import itertools

    temp_query_list = []
    query_list = []

    titles = match.match_dict.get('titles', 'none').split('\n(title change)')[0]  # trims the title change line off title field
    title_query = {'{}'.format(titles): {'titles': '{}'.format(titles)}}
    all_titles_query = {'all titles': {'titles': {'$ne': 'none'}}}
    no_titles_query = {'no title': {'titles': 'none'}}
    title_queries = []
    if title_query.values() == no_titles_query.values():
        title_queries.append(no_titles_query)
    else:
        title_queries.extend([title_query, all_titles_query])

    matchtype = match.match_dict.get('matchtype', 'normal')
    matchtype_monograms = matchtype.split()
    for index, monogram in enumerate(matchtype_monograms):
        matchtype_monograms[index] = '{} '.format(monogram)     # all need to end in spaces for next step to work right

    matchtype_ngrams = itertools.combinations(matchtype_monograms, r=len(matchtype_monograms))
    matchtype_queries = []
    for ngram in matchtype_ngrams:
        clean_ngram = ngram[0].strip()
        temp_query = {'matchtype contains \'{}\''.format(clean_ngram): {'matchtype': {'$regex': '{}'.format(clean_ngram)}}}
        matchtype_queries.append(temp_query)

    for t in title_queries:
        for t_name, t_query in t.items():
            for m in matchtype_queries:
                for m_name, m_query in m.items():
                    query_name = '{}, {}'.format(t_name, m_name)
                    and_query = {'$and': [t_query, m_query]}
                    combined_query = {query_name: and_query}
                    temp_query_list.append(combined_query)

    for query in temp_query_list:
        if query not in query_list:
            query_list.append(query)        # prevents processing duplicates

    temp_query_list.extend(title_queries)
    temp_query_list.extend(matchtype_queries)
    query_list.append({'Unfiltered': None})
    return query_list


def set_wintype_dicts(simple=False):
    if simple:
        wintype_dict = {
            "def. (pin)": 1,
            "(pin)": 1,
            "def. (sub)": 1,
            "(sub)": 1,
            "def. (dq)": 1,
            "(dq)": 1,
            "def. (forfeit)": 1,
            "def. (co)": 1,
            "def. (ko)": 1,
            "def. (tko)": 1,
            "def.": 1,

            "draw (nc)": 90,
            "draw (dco)": 90,
            "draw (time)": 90,
            "draw (ddq)": 90,
            "draw (dpin)": 90,
            "draw": 90,

            None: 99
        }

    else:
        wintype_dict = {
            "def. (pin)": 2,
            "(pin)": 2,
            "def. (sub)": 3,
            "(sub)": 3,
            "def. (dq)": 4,
            "(dq)": 4,
            "def. (forfeit)": 5,
            "def. (co)": 6,
            "def. (ko)": 7,
            "def. (tko)": 8,
            "def.": 1,

            "draw (nc)": 91,
            "draw (dco)": 92,
            "draw (time)": 93,
            "draw (ddq)": 94,
            "draw (dpin)": 95,
            "draw": 90,

            None: 99
        }

    return wintype_dict


def set_title_dicts(simple=False):
    if simple:
        title_dict = {
            'none': 0,
            'unknown title': 99
        }
    else:
        title_dict = {
            'none': 0,
            'money in the bank briefcase': 0,
            '(title change)': 0,
            'wwe king of the ring': 0,

            'wwe universal championship': 1,

            'wwe intercontinental championship': 2,

            'wwe raw womens title': 3,
            'wwe raw womens championship': 3,
            'wwe divas championship': 3,
            'wwe womens championship (2016)': 3,

            'wwe raw tag team championship': 4,
            'wwe raw tag championship': 4,
            'wwe tag team championship': 4,

            'wwe cruiserweight title': 4,
            'wwe cruiserweight championship': 4,
            'cruiserweight classic championship\nwwe cruiserweight title': 4,

            'wwe championship': 11,
            'wwe world heavyweight championship': 11,
            'wwe world championship': 11,
            'world heavyweight championship (wwe)': 11,

            'wwe united states championship': 12,

            'wwe smackdown womens championship': 13,
            'wwe smackdown womens title': 13,

            'wwe smackdown tag team championship': 14,
            'wwe smackdown tag championship': 14,

            'wwe united states championship\nwwe world heavyweight championship': 20,

            'nxt championship': 31,

            'mxt north american championship': 32,

            'wwe united kingdom championship': 33,

            'nxt womens title': 34,

            'nxt tag team titles': 35,
            'dusty rhodes tag team classic cup': 35,

            'unknown title': 99
        }

    return title_dict


def main():
    # the main goddamn program, duh
    print('\n\n\n\n\nk starting now.')

    global wintype_dict
    global title_dict
    global verbose

    verbose = False
    wintype_dict = set_wintype_dicts(simple=True)
    title_dict = set_title_dicts(simple=True)

    # wrestlemania 34 matches start here
    wrestlefestival = Event(event_dict={
        'name': 'wrestlemania 34',
        'date': 20180408,
        'matches': [
            {'name': 'universal championship',
                   'titles':'wwe universal championship',
                   'matchtype': 'normal',
                   'teams':[{'members': ['brock lesnar']},
                            {'members': ['roman reigns']}]},

            {'name': 'wwe championship',
                   'titles':'wwe championship',
                   'matchtype': 'normal',
                   'teams':[{'members': ['aj styles']},
                            {'members': ['shinsuke nakamura']}]},

            {'name': 'rousey debut',
                   'matchtype': 'tag',
                   'teams':[{'members': ['kurt angle']},     # rhonda rousey isn't in the db
                            {'members': ['stephanie mcmahon', 'triple h']}]},

            {'matchtype': 'tag',
                    'teams':[{'members': ['sami zayn', 'kevin owens']},
                             {'members': ['shane mcmahon', 'daniel bryan']}]},

            {'name': 'smackdown womens championship',
                   'titles':'wwe smackdown womens championship',
                   'teams':[{'members': ['charlotte flair']},
                            {'members': ['asuka']}]},

            {'name': 'raw womens championship',
                   'titles':'wwe raw womens championship',
                   'teams':[{'members': ['alexa bliss']},
                            {'members': ['nia jax']}]},

            {'name': 'intercontinental championship',
                   'titles':'wwe intercontinental championship',
                   'teams':[{'members': ['the miz']},
                            {'members': ['seth rollins']},
                            {'members': ['finn balor']}]},

            {'name': 'united states championship',
                   'titles':'wwe united states championship',
                   'teams':[{'members': ['randy orton']},
                            {'members': ['bobby roode']},
                            {'members': ['jinder mahal']},
                            {'members': ['rusev']}]},

            {'name': 'smackdown tag championship',
                   'titles':'wwe smackdown tag championship',
                   'matchtype': 'tag',
                   'teams':[
                       {'name': 'the usos',
                        'members':['jey uso', 'jimmy uso']},
                       {'name': 'the new day',
                        'members': ['big e', 'kofi kingston']},
                       {'name': 'the bludgeon brothers',
                        'members': ['harper', 'rowan']}]},

            {'name': 'raw tag championship',
                   'titles':'wwe raw tag championship',
                   'matchtype': 'tag',
                   'teams':[{'name': 'the bar',
                             'members': ['cesaro', 'sheamus']},
                            {'members': ['braun strowman']}  # unannounced teammate
                           ]},

            {'name': 'cruiserweight championship',
                   'titles':'wwe cruiserweight championship',
                   'teams':[{'members': ['cedric alexander']},
                            {'members': ['mustafa ali']}]},

            {'name': 'mens battle royale',
                   'matchtype': 'battle royale',
                   'teams':[
                            {'members': ['Aiden English']},
                            {'members': ['Konnor']},
                            {'members': ['Curt Hawkins']},
                            {'members': ['R-Truth']},
                            {'members': ['Primo Colon']},
                            {'members': ['Mike Kanellis']},
                            {'members': ['Tyler Breeze']},
                            {'members': ['Viktor']},
                            {'members': ['Zack Ryder']},
                            {'members': ['Karl Anderson']},
                            {'members': ['Luke Gallows']},
                            {'members': ['Apollo']},
                            {'members': ['Shelton Benjamin']},
                            {'members': ['Rhyno']},
                            {'members': ['Dash Wilder']},
                            {'members': ['Scott Dawson']},
                            {'members': ['Bo Dallas']},
                            {'members': ['Curtis Axel']},
                            {'members': ['Sin Cara']},
                            {'members': ['Fandango']},
                            {'members': ['Heath Slater']},
                            {'members': ['Chad Gable']},
                            {'members': ['Titus ONeil']},
                            {'members': ['Goldust']},
                            {'members': ['Tye Dillinger']},
                            {'members': ['Dolph Ziggler']},
                            {'members': ['Kane']},
                            {'members': ['Mojo Rawley']},
                            {'members': ['Baron Corbin']},
                            {'members': ['matt hardy']}]},

            {'name': 'womens battle royale',
                   'matchtype': 'womens battle royale',
                   'teams':[
                       {'members': ['Carmella']},
                       {'members': ['Dana Brooke']},
                       {'members': ['Mandy Rose']},
                       {'members': ['Sonya Deville']},
                       {'members': ['Kairi Sane']},
                       {'members': ['Lana']},
                       {'members': ['Kavita Devi']},
                       {'members': ['Taynara Conti']},
                       {'members': ['Bianca Belair']},
                       {'members': ['Dakota Kai']},
                       {'members': ['Becky Lynch']},
                       {'members': ['Mickie James']},
                       {'members': ['Peyton Royce']},
                       {'members': ['Natalya']},
                       {'members': ['Liv Morgan']},
                       {'members': ['Ruby Riott']},
                       {'members': ['Sarah Logan']},
                       {'members': ['Sasha Banks']},
                       {'members': ['Bayley']},
                       {'members': ['naomi']}]}
    ]})

    dataset_dict = {}
    model_dict = {}

    dataset_minimum = 100

    for match in wrestlefestival.event_dict['matches']:
        query_list = generate_queries(match)
        match_dataset_list = []
        match_model_list = []

        for item in query_list:
            for name, query in item.items():
                if name in dataset_dict.keys():
                    match_dataset_list.append(dataset_dict[name])
                else:
                    dataset_dict[name] = Dataset(limit=9000, query=query, name=name, verbose=verbose)
                    match_dataset_list.append(dataset_dict[name])

                if name in model_dict.keys():
                    match_model_list.append(model_dict[name])
                else:
                    if dataset_dict[name].train_dataset.shape[0] < dataset_minimum:
                        if verbose:
                            print('dataset \'{}\' under minimum number of entries; skipping model build.\n'.format(name))
                        model_dict[name] = None
                        match_model_list.append(model_dict[name])
                    else:
                        model_dict[name] = Model(train_steps=2500, model_type='hybrid', dataset=dataset_dict[name], name=name, layer_specs=[80, 90, 100])
                        match_model_list.append(model_dict[name])

        prediction = match.aggregate_predict(model_list=match_model_list)

        print(prediction['prediction_string'].format(**prediction))


def test():
    # catch-all function for one-off testing
    pass


if __name__ == '__main__':
    test()
    main()