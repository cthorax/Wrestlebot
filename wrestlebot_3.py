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

title_dict = {
    'none': 0,
    'money in the bank briefcase': 0,
    '(title change)': 0,
    'wwe king of the ring': 0,

    'wwe universal championship': 1,

    'wwe intercontinental championship': 2,
    
    'wwe raw womens title': 3,
    'wwe divas championship': 3,
    'wwe womens championship (2016)': 3,

    'wwe raw tag team championship': 4,
    'wwe tag team championship': 4,

    'wwe cruiserweight title': 4,
    'cruiserweight classic championship\nwwe cruiserweight title': 4,

    'wwe championship': 11,
    'wwe world heavyweight championship': 11,
    'wwe world championship': 11,
    'world heavyweight championship (wwe)': 11,

    'wwe united states championship': 12,

    'wwe smackdown womens championship': 13,
    'wwe smackdown womens title': 13,
    
    'wwe smackdown tag team championship': 14,

    'wwe united states championship\nwwe world heavyweight championship': 20,

    
    'nxt championship': 31,

    'wwe united kingdom championship': 32,

    'nxt womens title': 33,

    'nxt tag team titles': 34,
    'dusty rhodes tag team classic cup': 34,
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
    32: 'wwe uk title',
    33: 'nxt womens title',
    34: 'nxt tag team title',
}


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
        'wintype']
    return pd.DataFrame(columns=columns)


class Wrestler(object):
    def __init__(self, name, train=False, date=30001231, source_match=''):
        assert isinstance(date, int)

        self.source_match = source_match
        self.is_training = train
        self.circa = date
        self.name = name.lower()
        self.history = history_dataset()
        self.update()

    def update(self):
        # this will update all relevant wrestler stats when called
        try:
            aliases = list(wrestler_collection.find({"name": self.name}))
            assert aliases  # checks the DB for an instance of the wrestler, and fails if none exists
            self.aliases = aliases[0]['name']
            self.get_history()
        except AssertionError:
            # deletion doesn't work since this is done in a reference to itself, so clearing all values and removing Nones later will have to suffice
            print("wrestler '{}' does not exist.".format(self.name))
            self.source_match = None
            self.is_training = None
            self.circa = None
            self.name = ''
            self.history = None

    def get_history(self):
        # changes list from MongoDB results into Dataframe from pandas

        if self.is_training:
            number_of_matches = 11
        else:
            number_of_matches = 10

        history = []
        for alias in self.aliases:
            temp_history = list(match_collection.find({'$and': [
                {"_id": {'$ne': self.source_match}},
                {"wrestlers": alias},
                {"date": {'$lt': self.circa}}
            ]
            }).limit(number_of_matches).sort("date", pymongo.DESCENDING))
            history.extend(temp_history)
        history = sorted(history, key=lambda x: x['date'], reverse=True)[:number_of_matches]
        # sorts the list by date descending and then truncates to number_of_matches entries
        history = sorted(history, key=lambda x: x['date'], reverse=False)
        # sorts it back to date ascending. since the optional predictor values will be at the front,
        # we need to work in reverse as we build.

        history_dataframe = history_dataset()

        match_number = len(history) -1 -self.is_training   # minus one due to zero index, minus one due to wintype label being unnumbered
        blank_dict = {
                '0_duration': [0],
                '0_number_of_teams': [0],
                '0_opponents': [0],
                '0_allies': [0],
                '1_duration': [0],
                '1_number_of_teams': [0],
                '1_opponents': [0],
                '1_allies': [0],
                '2_duration': [0],
                '2_number_of_teams': [0],
                '2_opponents': [0],
                '2_allies': [0],
                '3_duration': [0],
                '3_number_of_teams': [0],
                '3_opponents': [0],
                '3_allies': [0],
                '4_duration': [0],
                '4_number_of_teams': [0],
                '4_opponents': [0],
                '4_allies': [0],
                '5_duration': [0],
                '5_number_of_teams': [0],
                '5_opponents': [0],
                '5_allies': [0],
                '6_duration': [0],
                '6_number_of_teams': [0],
                '6_opponents': [0],
                '6_allies': [0],
                '7_duration': [0],
                '7_number_of_teams': [0],
                '7_opponents': [0],
                '7_allies': [0],
                '8_duration': [0],
                '8_number_of_teams': [0],
                '8_opponents': [0],
                '8_allies': [0],
                '9_duration': [0],
                '9_number_of_teams': [0],
                '9_opponents': [0],
                '9_allies': [0],

                '0_wintype': [99],
                '1_wintype': [99],
                '2_wintype': [99],
                '3_wintype': [99],
                '4_wintype': [99],
                '5_wintype': [99],
                '6_wintype': [99],
                '7_wintype': [99],
                '8_wintype': [99],
                '9_wintype': [99],

                '0_titles': [0],
                '1_titles': [0],
                '2_titles': [0],
                '3_titles': [0],
                '4_titles': [0],
                '5_titles': [0],
                '6_titles': [0],
                '7_titles': [0],
                '8_titles': [0],
                '9_titles': [0],

                '0_matchtype': [7438723275317868808],
                '1_matchtype': [7438723275317868808],
                '2_matchtype': [7438723275317868808],
                '3_matchtype': [7438723275317868808],
                '4_matchtype': [7438723275317868808],
                '5_matchtype': [7438723275317868808],
                '6_matchtype': [7438723275317868808],
                '7_matchtype': [7438723275317868808],
                '8_matchtype': [7438723275317868808],
                '9_matchtype': [7438723275317868808],

                'wintype': [99]
            }
        blank_dataframe = pd.DataFrame.from_dict(blank_dict)

        temp_dict = {}
        for match in history:
            for alias in self.aliases:
                if alias in match['wrestlers']:
                    current_alias = alias
                    break

            is_winner = current_alias in match['winners']

            del match['date']
            del match['card']
            del match['wrestlers']
            del match['winners']
            del match['_id']

            if match_number <= -1:
                del match['teams']
                del match['matchtype']
                del match['duration']
                del match['titles']

            for key, value in match.items():
                if key == 'teams':
                    number_of_teams = len(value)
                    opponents = 0
                    allies = 0
                    for team in value:
                        if current_alias in team:
                            allies = len(team) - 1
                        else:
                            opponents += len(team)

                    temp_dict["".join([str(match_number), "_number_of_teams"])] = [number_of_teams]
                    temp_dict["".join([str(match_number), "_opponents"])] = [opponents]
                    temp_dict["".join([str(match_number), "_allies"])] = [allies]

                elif key == 'titles':
                    if value:
                        value = value.split('\n(title change)')[0]  # trims the title change line off title field
                        temp_dict["".join([str(match_number), "_titles"])] = [title_dict[value]]
                    else:
                        temp_dict["".join([str(match_number), "_titles"])] = [0]

                elif key == 'matchtype':
                    if value.strip():   # don't remember if blank matchtypes are '' or ' '.
                        temp_dict["".join([str(match_number), "_matchtype"])] = [hash(value)]
                    else:
                        temp_dict["".join([str(match_number), "_matchtype"])] = [hash('normal')]

                elif key == 'wintype':
                    adjusted_value = wintype_dict[value]
                    if adjusted_value >= 90 or is_winner:
                        pass
                    else:
                        adjusted_value += 50

                    if match_number == -1:
                        temp_dict["wintype"] = [adjusted_value]   # labels need to be numbers, not string
                    else:
                        temp_dict["".join([str(match_number), "_wintype"])] = [adjusted_value]

                else:
                    temp_dict["".join([str(match_number), "_", key])] = [value]

            match_number -= 1

        temp_dataframe = pd.DataFrame.from_dict(temp_dict)
        history_dataframe = history_dataframe.combine_first(temp_dataframe)
        history_dataframe = history_dataframe.combine_first(blank_dataframe)    # fills in any remaining holes from temp_dataframe with the blank values

        self.history = history_dataframe.astype(int)


class Team(object):
    def __init__(self, team_dict=None):
        if team_dict:
            try:
                self.team_name = team_dict['name']
            except KeyError:
                self.team_name = 'empty team'

            try:
                self.members = team_dict['members']
                self.update()
            except KeyError:
                self.members = []

        else:
            self.members = []
            self.team_name = 'empty team'

    def add_members(self, new_members):
        if isinstance(new_members, list):
            self.members.extend(new_members)

        elif isinstance(new_members, Wrestler):
            if new_members.name:
                self.members.append(new_members)

        self.update()

    def update(self):
        # this will update all relevant team stats when called.
        for index, member in enumerate(self.members):
            if isinstance(member, Wrestler):
                pass
            elif isinstance(member, str):
                temp_member = Wrestler(name=member)
                self.members[index] = temp_member

        if self.team_name == 'empty team' or not self.team_name:
            team_name = ""

            for wrestler in self.members:
                team_name = ", ".join([team_name, wrestler.name])

            team_name = team_name[2:]  # removes extra ', ' from front
            team_name = ', & '.join(team_name.rsplit(', ',1))
            if team_name.count(',') == 1:  # oxford comma fixer
                team_name = team_name.replace(',', '')

            self.team_name = team_name


class Match(object):
    def __init__(self, match_dict=None):
        if match_dict:
            self.teams = match_dict['teams']
            try:
                self.match_name = match_dict['name']
            except KeyError:
                self.match_name = 'empty match'
            try:
                self.match_type = match_dict['matchtype']
            except KeyError:
                self.match_type = 'normal'
            try:
                self.titles = match_dict['titles']
            except KeyError:
                self.titles = 'none'
            self.update()
        else:
            self.teams = []
            self.match_type = 'normal'
            self.titles = 'none'
            self.match_name = 'empty match'

    def add_teams(self, new_teams):
        if isinstance(new_teams, list):
            self.teams.extend(new_teams)
        else:
            self.teams.append(new_teams)

    def update(self):
        #this will update all relevant Match stats when called.
        if self.teams:
            for index, team in enumerate(self.teams):
                if isinstance(team, Team):
                    pass
                elif isinstance(team, dict):
                    temp_team = Team(team_dict=team)
                    self.teams[index] = temp_team

                elif isinstance(team, list):
                    team_dict = {
                        'name': None,
                        'members': team
                    }
                    temp_team = Team(team_dict=team_dict)
                    self.teams[index] = temp_team


        if not self.match_name or self.match_name == 'empty match':
            match_name = ""

            for team in self.teams:
                match_name = " VS ".join([match_name, team.team_name])

            self.match_name = match_name[4:]  # removes extra ' VS ' from front


    def predict(self, model):
        assert isinstance(model, Model)

        self.predictions = None
        # create the team predictions from the wrestler predictions
        for team in self.teams:
            team.predictions = []
            team.naive_win = []
            team.naive_lose = []
            team.naive_draw = []

            for wrestler in team.members:
                dataset = wrestler.history.copy()
                try:
                    dataset.pop('wintype')
                except IndexError:
                    pass

                wrestler.predictions = model.make_prediction(dataset)
                wrestler.win = np.sum(wrestler.predictions[1:8])
                wrestler.lose = np.sum(wrestler.predictions[51:58])
                wrestler.draw = np.sum(wrestler.predictions[91:95])

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
        for team in self.teams:
            win_product.append(team.naive_win)
            lose_product.append(team.naive_lose)
            draw_product.append(team.naive_draw)

        win_product = np.prod(win_product)
        lose_product = np.prod(lose_product)
        draw_product = np.prod(draw_product)

        self.predicted_winner = None
        self.predicted_wintype = None
        norm = 0
        for team in self.teams:
            team.true_win = np.true_divide(lose_product, team.naive_lose)
            team.true_win = np.prod([team.true_win, team.naive_win])

            team.true_lose = np.true_divide(win_product, team.naive_win)
            team.true_lose = np.prod([team.true_lose, team.naive_lose])

            team.true_draw = draw_product
            norm = np.sum([team.true_win, norm])

        norm = np.sum([team.true_draw, norm])     # true_draw should be the same for all teams, so this only needs to be added once.

        if self.predicted_winner is None:
            self.predicted_winner = [team.team_name, team.true_win]

        if self.predicted_winner[1] < team.true_win:      # seperate from above bc None is not subscriptable, so attempting to find None[1] in the OR statement will make it crash.
            self.predicted_winner = [team.team_name, team.true_win]

            if self.predicted_winner[0] == team.team_name:
                for wintype, type_probability in enumerate(team.predictions[0:8]):
                    if type_probability == max(team.predictions):
                        type_probability = np.true_divide(type_probability, team.naive_win)
                        type_probability = np.prod([type_probability, team.true_win])
                        self.predicted_wintype = [wintype, type_probability]

        self.predicted_winner[1] = np.true_divide(self.predicted_winner[1], norm)

        print(
            "using {} model:\n\npredicted winner of '{}' is\n{} ({:.3%}).\nMost likely is {} ({:.3%})".format(
            model.model_type,
            self.match_name,
            self.predicted_winner[0],
            self.predicted_winner[1],
            wintype_antidict[self.predicted_wintype[0]],
            self.predicted_wintype[1]
            )
        )


class Dataset(object):
    def __init__(self, limit=None, query=None, backup_every=50):
        assert isinstance(limit, int) or not limit

        self.train_dataset = history_dataset()
        train_backup_file = 'train - {}.csv'.format(query).replace(':', ';')
        train_offset = 0

        self.test_dataset = history_dataset()
        test_backup_file = 'test - {}.csv'.format(query).replace(':', ';')
        test_offset = 0

        self.validate_dataset = history_dataset()
        validate_backup_file = 'validate - {}.csv'.format(query).replace(':', ';')
        validate_offset = 0

        for dataset, backup_file, offset in [(self.train_dataset, train_backup_file, train_offset), (self.test_dataset, test_backup_file, test_offset), (self.validate_dataset, validate_backup_file, validate_offset)]:
            try:
                with open(backup_file, 'r') as dataset_backup:
                    dataset = dataset.append(pd.read_csv(dataset_backup))
                    offset = dataset.shape[0]   # to avoid dataset inflation.
            except FileNotFoundError as e:
                print('no backup file found, starting from scratch.')
            except BaseException as e: # not sure what error yet
                print('file is corrupted. removing.\n', e)
                os.remove(backup_file)

        if limit:
            size = limit // 3
            # if the backup files for some but not all of the sets are over the limit, don't truncate to save loading.
            true_limit = limit - min(train_offset, size) - min(test_offset, size) - min(validate_offset, size)
            if true_limit > 0:
                if query:
                    results = list(match_collection.aggregate([
                        {'$match': query},
                        {'$sample': {'size': true_limit}}
                    ]))
                else:
                    results = list(match_collection.aggregate([
                        {'$sample': {'size': true_limit}}
                    ]))

            else:
                results = []
                print('\'{}\' datasets restored from file.'.format(self.dataset_name))

        else:
            if query:
                results = list(match_collection.find(query))
            else:
                results = list(match_collection.find())
            size = len(results) // 3

        random.shuffle(results)
        test_results = results[0:size]
        train_results = results[size + 1: 2*size]
        validate_results = results[2*size + 1: 3*size]

        for results, dataset, backup_file in [(train_results, self.train_dataset, train_backup_file), (test_results, self.test_dataset, test_backup_file), (validate_results, self.validate_dataset, validate_backup_file)]:
            for index, match in enumerate(results):
                print('\rstarting \'{}\' dataset: item {} of {}'.format(self.dataset_name, index+1+offset, len(results)+offset), end='')
                for name in match['wrestlers']:
                    temp_wrestler = Wrestler(name=name, train=True, date=match['date'], source_match=match['_id'])
                    dataset = dataset.append(temp_wrestler.history, ignore_index=True)
                if index % backup_every == 0:
                    with open(backup_file, 'w') as dataset_backup:
                        dataset.to_csv(dataset_backup, index=False, header=True)
            print('\r\'{}\' dataset finished.'.format(backup_file[:-4]))


class Model(object):
    def __init__(self, batch_size=100, train_steps=1000, model_type='linear', dataset=None, layer_specs=[10, 10, 10]):
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

        # get datasets to train and test
        (self.train_x, self.train_y), (self.test_x, self.test_y), (self.validate_x, self.validate_y) = self.load_data()

        # train model
        self.train_model()

        # evaluate model
        self.assess_model()

    def load_data(self, y_name="wintype"):     # when no longer testing, change limit probably
        train = self.train_dataset
        test = self.test_dataset
        validate = self.validate_dataset

        # right now this only works for numeric values
        train_x, train_y = train.astype(int), train.get(y_name).astype(int)
        test_x, test_y = test.astype(int), test.get(y_name).astype(int)
        validate_x, validate_y = validate.astype(int), validate.get(y_name).astype(int)

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

    def train_model(self):
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
            '9_duration', '9_number_of_teams', '9_opponents', '9_allies'
        ]
        for key in numeric_columns:
            numeric_feature_columns.append(
                tf.feature_column.numeric_column(key=key)
            )

        win_columns = [
            '0_wintype', '1_wintype', '2_wintype', '3_wintype', '4_wintype',
            '5_wintype', '6_wintype', '7_wintype', '8_wintype', '9_wintype'
        ]
        title_columns = [
            '0_titles', '1_titles', '2_titles', '3_titles', '4_titles',
            '5_titles', '6_titles', '7_titles', '8_titles', '9_titles'
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
            '5_matchtype', '6_matchtype', '7_matchtype', '9_matchtype'
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
                model_dir='.\wrestlebot linear model',
                feature_columns=numeric_feature_columns + wide_categorical_columns,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'deep':
            classifier = tf.estimator.DNNClassifier(
                model_dir='.\wrestlebot deep model with {}'.format(self.layer_specs),
                feature_columns=numeric_feature_columns + deep_categorical_columns,
                hidden_units=self.layer_specs,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'hybrid':
            classifier = tf.estimator.DNNLinearCombinedClassifier(
                model_dir='.\wrestlebot hybrid model with {}'.format(self.layer_specs),
                linear_feature_columns=numeric_feature_columns + wide_categorical_columns,
                dnn_feature_columns=numeric_feature_columns + deep_categorical_columns,
                dnn_hidden_units=self.layer_specs,
                n_classes=max(wintype_antidict.keys()) + 1,  # labels must be strictly less than classes
            )

        # Train the Model.
        print('training the {} classifier for {} steps.'.format(self.model_type, self.train_steps))
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

    def assess_model(self):
        # Evaluate the model.
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.eval_input_fn(
                self.test_x,
                self.test_y
        ))

        print('\nTest set accuracy for model: {accuracy:.3%}\n'.format(**eval_result))
        self.model_accuracy = eval_result['accuracy']

    def compare(self):
        # Evaluate the model based on the validate set to compare two models.
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.eval_input_fn(
                self.validate_x,
                self.validate_y
        ))

        return eval_result['accuracy']


def compare_models(model_list):
    assert isinstance(model_list, list)
    assert isinstance(model_list[0], Model)
    assert isinstance(model_list[1], Model)

    best = 0
    winner = None
    for index, model in enumerate(model_list):
        result = model.compare()
        if result > best:
            best = result
            winner = index

    print()
    return model_list[winner]


def main():
    # the main goddamn program, duh
    print('\n\n\n\n\nk starting now.')

    # wrestlemania 34 matches start here
    wrestlefestival = [
        Match({'name': 'universal championship',
               'title': 'universal championship',
               'matchtype': 'normal',
               'teams':[['brock lesnar'],
                        ['roman reigns']]}),

        Match({'name': 'wwe championship',
               'title': 'wwe championship',
               'matchtype': 'normal',
               'teams':[['aj styles'],
                        ['shinsuke nakamura']]}),

        Match({'name': 'rousey debut',
               'matchtype': 'tag',
               'teams':[['kurt angle'],     # rhonda rousey isn't in the db
                        ['stephanie mcmahon', 'triple h']]}),

        Match({ 'matchtype': 'tag',
                'teams':[['sami zayn', 'kevin owens'],
                        ['shane mcmahon', 'daniel bryan']]}),

        Match({'name': 'smackdown womens championship',
               'title': 'smackdown womens championship',
               'teams':[['charlotte flair'],
                        ['asuka']]}),

        Match({'name': 'raw womens championship',
               'title': 'raw womens championship',
               'teams':[['alexa bliss'],
                        ['nia jax']]}),

        Match({'name': 'intercontinental championship',
               'title': 'intercontinental championship',
               'teams':[['the miz'],
                        ['seth rollins'],
                        ['finn balor']]}),

        Match({'name': 'united states championship',
               'title': 'united states championship',
               'teams':[['randy orton'],
                        ['bobby roode'],
                        ['jinder mahal'],
                        ['rusev']]}),

        Match({'name': 'smackdown tag championship',
               'title': 'smackdown tag championship',
               'matchtype': 'tag',
               'teams':[
                   {'name': 'the usos',
                    'members':['jey uso', 'jimmy uso']},
                   {'name': 'the new day',
                    'members': ['big e', 'kofi kingston', 'xavier woods']},  # likely only 2
                   {'name': 'the bludgeon brothers',
                    'members': ['harper', 'rowan']}]}),

        Match({'name': 'raw tag championship',
               'title': 'raw tag championship',
               'matchtype': 'tag',
               'teams':[{'name': 'the bar',
                         'members': ['cesaro', 'sheamus']},
                        ['braun strowman']  # unannounced teammate
                       ]}),

        Match({'name': 'cruiserweight championship',
               'title': 'cruiserweight championship',
               'teams':[['cedric alexander'],
                        ['mustafa ali']]}),

        Match({'name': 'mens battle royale',
               'teams':[

               ]}),

        Match({'name': 'womens battle royale',
               'teams':[

               ]})
     ]

    unfiltered_dataset = Dataset(limit=5000)
    normal_dataset = Dataset(limit=5000, query={'matchtype': 'normal'})
    normal_title_dataset = Dataset(limit=5000, query={'$and': [{'titles': 'none'}, {'matchtype': 'normal'}]})
    normal_no_title_dataset = Dataset(limit=5000, query={'$and': [{'titles': {'$ne': 'none'}}, {'matchtype': 'normal'}]})
    title_match_dataset = Dataset(limit=5000, query={'titles': {'ne': 'none'}})
    no_title_dataset = Dataset(limit=5000, query={'titles': 'none'})
    tag_team_dataset = Dataset(limit=5000, query={'matchtype': {'$regex': 'tag'}})
    tag_no_title_dataset = Dataset(limit=5000, query={'$and': [{'titles': 'none'}, {'matchtype': {'$regex': 'tag'}}]})
    tag_title_dataset = Dataset(limit=5000, query={'$and': [{'titles': {'$ne': 'none'}}, {'matchtype': {'$regex': 'tag'}}]})

    # linear = Model(train_steps=1000, model_type='linear', dataset=dataset)
    # deep = Model(train_steps=1000, model_type='deep', dataset=dataset, layer_specs=[88, 88])
    hybrid = Model(train_steps=1000, model_type='hybrid', dataset=unfiltered_dataset, layer_specs=[80, 90, 100])
    hybrid_normal = Model(train_steps=1000, model_type='hybrid', dataset=normal_dataset, layer_specs=[80, 90, 100])
    hybrid_title = Model(train_steps=1000, model_type='hybrid', dataset=title_match_dataset, layer_specs=[80, 90, 100])
    hybrid_no_title = Model(train_steps=1000, model_type='hybrid', dataset=no_title_dataset, layer_specs=[80, 90, 100])
    hybrid_tag = Model(train_steps=1000, model_type='hybrid', dataset=tag_team_dataset, layer_specs=[80, 90, 100])
    hybrid_tag_no_title = Model(train_steps=1000, model_type='hybrid', dataset=tag_no_title_dataset, layer_specs=[80, 90, 100])
    hybrid_tag_title = Model(train_steps=1000, model_type='hybrid', dataset=tag_title_dataset, layer_specs=[80, 90, 100])
    hybrid_normal_title = Model(train_steps=1000, model_type='hybrid', dataset=normal_title_dataset, layer_specs=[80, 90, 100])
    hybrid_normal_no_title = Model(train_steps=1000, model_type='hybrid', dataset=normal_no_title_dataset, layer_specs=[80, 90, 100])

    for match in wrestlefestival:
        if match.title == 'none' and match.matchtype == 'normal':
            model_list = [hybrid_normal_no_title, hybrid_normal, hybrid]
        elif not match.title == 'none' and match.matchtype == 'normal':
            model_list = [hybrid_normal_title, hybrid_normal, hybrid_title, hybrid]
        elif match.title == 'none' and match.matchtype == 'tag':
            model_list = [hybrid_tag_no_title, hybrid_tag, hybrid]
        elif not match.title == 'none' and match.matchtype == 'tag':
            model_list = [hybrid_tag_title, hybrid_tag, hybrid]
        winner = compare_models(model_list=model_list)
        match.predict(winner)
    pass


def test():
    # catch-all function for one-off testing
    pass


if __name__ == '__main__':
    test()
    main()