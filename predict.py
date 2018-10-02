
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
number_of_history_matches = int(config['all files']['number_of_history_matches'])
max_number_of_wrestlers = int(config['all files']['max_number_of_wrestlers'])
path_to_tf_files = config['predict only']['path_to_tf_files']
verbose = config['all files'].getboolean('verbose')

db_url = config['all files']['db_url']
db = records.Database(db_url=db_url)


def competitor_string_to_list(competitor_string):
    competitor_list = []
    for team in competitor_string.split(team_end_marker)[:-1]:
        competitor_list.append(team.replace(wrestler_start_marker, '').split(wrestler_end_marker)[:-1])
    
    return competitor_list


def competitor_list_to_dict(competitor_list):
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


def prediction_series_dict():
    wrestler_index_template_list = ['id', 'dob', 'nationality']
    history_match_index_template_list = ['days_since_match', 'wintype', 'title', 'matchtype', 'opponents', 'allies', 'location', 'attendance', 'promotion', 'ppv']
    index_dict = {'current_title': 0, 'current_matchtype': 0, 'winner': -1}

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


def make_prediction_match_dict(id, index_dict, db=records.Database(db_url=db_url), event_date=29991231):
    matches_query_string = """
SELECT m.id, m.date, m.wintype, m.titles, m.matchtype, t.competitors
FROM match_table m JOIN team_table t ON m.match_id = t.match_id
WHERE date < {event} AND m.match_id IN (
    SELECT match_id FROM team_table WHERE competitors LIKE '%{start}{id}{end}%'
)
ORDER BY date DESC
LIMIT {limit}"""

    matches_query = db.query(matches_query_string.format(
        event=event_date, start=wrestler_start_marker, id=id, end=wrestler_end_marker, limit=number_of_history_matches+1
    )).as_dict()
    current_match = matches_query.pop()
    match_rowid = current_match['id']
    index_dict['current_wintype'] = current_match['wintype']
    index_dict['current_titles'] = current_match['titles']
    index_dict['current_matchtype'] = current_match['matchtype']

    current_match_competitors_list = competitor_string_to_list(current_match['competitors'])
    current_match_competitor_dict = competitor_list_to_dict(current_match_competitors_list)
    for wrestler_number in range(max_number_of_wrestlers):
        if current_match_competitor_dict[wrestler_number] == 0:
            break
        for team in current_match_competitors_list:
            if current_match_competitor_dict[wrestler_number] in team:
                index_dict['w{wn}_current_allies'.format(wn=wrestler_number)] = len(team) - 1
            else:
                index_dict['w{wn}_current_opponents'.format(wn=wrestler_number)] += len(team)

    for match_number, match in enumerate(matches_query):
        competitor_list = competitor_string_to_list(match['competitors'])
        competitor_dict = competitor_list_to_dict(competitor_list)
        for wrestler_number in range(max_number_of_wrestlers):
            if competitor_dict[wrestler_number] == 0:
                break
            else:
                index_dict['w{wn}m{mn}_id'.format(wn=wrestler_number, mn=match_number)] = competitor_dict[wrestler_number]
                stats_query = db.query("SELECT dob, nationality FROM wrestler_table WHERE id = '{id}'".format(id=competitor_dict[wrestler_number])).as_dict()
                for stats in stats_query:
                    index_dict['w{wn}m{mn}_dob'.format(wn=wrestler_number, mn=match_number)] = int(stats.get('dob', 0))
                    index_dict['w{wn}m{mn}_nationality'.format(wn=wrestler_number, mn=match_number)] = int(stats.get('nationality', 0))
                    index_dict['w{wn}m{mn}_days_since_match'.format(wn=wrestler_number, mn=match_number)] = event_date - match['date']
                    index_dict['w{wn}m{mn}_wintype'.format(wn=wrestler_number, mn=match_number)] = match['wintype']
                    index_dict['w{wn}m{mn}_title'.format(wn=wrestler_number, mn=match_number)] = match['titles']
                    index_dict['w{wn}m{mn}_matchtype'.format(wn=wrestler_number, mn=match_number)] = match['matchtype']

                for team in competitor_list:
                    if competitor_dict[wrestler_number] in team:
                        index_dict['w{wn}m{mn}_allies'.format(wn=wrestler_number, mn=match_number)] = len(team) - 1
                    else:
                        index_dict['w{wn}m{mn}_opponents'.format(wn=wrestler_number, mn=match_number)] += len(team)

                if id in competitor_list[0] and index_dict['winner'] == -1:
                    index_dict['winner'] = wrestler_number

    return match_rowid, index_dict


def make_dataset_dict(db=records.Database(db_url=db_url), number_of_matches=1000):
    dataset_dict = {'test': None, 'train': None, 'validate': None}
    blank_dict = prediction_series_dict()

    for key in dataset_dict.keys():
        if verbose:
            print("making '{key}' dataset from {matches} matches.".format(key=key, matches=number_of_matches))
        match_query = db.query("""
SELECT m.date AS date, t.competitors AS competitors
FROM match_table m JOIN team_table t ON t.match_id = m.match_id
WHERE id IN (
SELECT id FROM match_table ORDER BY RANDOM() LIMIT {limit}
)""".format(limit=number_of_matches)).as_dict()

        dict_of_match_dicts = {}
        for matchnum, match in enumerate(match_query, start=1):
            if verbose:
                print("\rprocessing match {num} of {maxnum}".format(num=matchnum, maxnum=number_of_matches), end='')
            teams_list = competitor_string_to_list(match['competitors'])
            for team in teams_list:
                for id in team:
                    temp_rowid, temp_dict = make_prediction_match_dict(id=id, index_dict=blank_dict.copy(), db=db, event_date=match['date'] + 1)
                    dict_of_match_dicts[temp_rowid] = temp_dict

        if verbose:
            print('... done!\ncombining datasets.')
        combined_dataset = pd.DataFrame(list(dict_of_match_dicts.values()), dtype='int', index=pd.Index(list(dict_of_match_dicts.keys())))
        combined_dataset.fillna(0, inplace=True)

        dataset_dict[key] = combined_dataset
        
    return dataset_dict

    
class Model(object):
    def __init__(self, batch_size=100, train_steps=1000, model_type='linear', dataset_dict=None, layer_specs=[10, 10, 10], name=None):
        """
        Called when Model is instantiated.
        :param batch_size: integer. pd.Dataframes used in training model are chopped up to contain this many rows.
        :param train_steps: integer. training process repeated this many times.
        :param model_type: string. linear, deep, or hybrid.
                                A linear model is essentially a linear regression.
                                A deep model is a fully connected NN, following layer specs.
                                A hybrid model has both a linear portion and a deep portion.
        :param dataset_dict: dict. in one of two forms; either a dict with training dataframes for keys 'train', 'test', and 'assess',
                                or a dict with only the key 'matches' and an int value to override the default number_of_matches peram
                                when make_dataset_dict is called. 
        :param layer_specs: list. each entry is the number of nodes in a layer, the number of elements in the list is the number of layers.
                                this param is ignored if model_type is 'linear'
        :param name: string. an optional name for the model for use when multiple models are served at once.
        """
        if verbose:
            print("initialising {type} model '{name}' with batch size {batch}, train steps {train}, and layer specs {layers}".format(
                type=model_type, name=name, train=train_steps, batch=batch_size, layers=layer_specs
            ))
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.model_type = model_type
        self.layer_specs = layer_specs
        self.name = name
        self.save_hash = hash('{} - {}'.format(self.name, self.layer_specs))
        
        if dataset_dict is None:
            dataset_dict = make_dataset_dict()
        elif list(dataset_dict.keys()) == ['matches']:
            dataset_dict = make_dataset_dict(number_of_matches=dataset_dict['matches'])
        self.test_dataset = dataset_dict['test']
        self.train_dataset = dataset_dict['train']
        self.validate_dataset = dataset_dict['validate']
        
        # get datasets to train and test
        (self.train_x, self.train_y), (self.test_x, self.test_y), (self.validate_x, self.validate_y) = self.load_data()

        # train model
        self.train_model()

        # evaluate model
        self.assess_model()

    def load_data(self, y_name="winner"):
        # right now this only works for numeric values
        train_x = self.train_dataset
        train_x = train_x.astype(int)
        train_y = self.train_dataset.get(y_name).astype(int)

        test_x = self.test_dataset.astype(int)
        test_y = self.test_dataset.get(y_name).astype(int)

        validate_x = self.validate_dataset.astype(int)
        validate_y = self.validate_dataset.get(y_name).astype(int)

        return (train_x, train_y), (test_x, test_y), (validate_x, validate_y)

    def train_input_fn(self, features, labels):
        # stolen from the iris_data tutorial
        """An input function for training"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)

        # Return the dataset.
        return dataset

    def eval_input_fn(self, features, labels):
        # stolen from the iris_data tutorial
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

    def train_model(self, verbose=False):
        # calls input_to_model to train a new model
        if verbose:
            print("training model for {train} steps".format(train=self.train_steps))

        # define feature columns
        numeric_feature_columns = []
        wide_categorical_columns = []
        deep_categorical_columns = []

        index_dict_keys = list(prediction_series_dict().keys())
        
        numeric_markers = ['dob', 'days_since_match', 'opponents', 'allies', 'attendance']
        categorical_markers = ['id', 'nationality', 'wintype', 'title', 'matchtype', 'winner', 'promotion','location', 'ppv']
        
        for key in index_dict_keys:
            for marker in numeric_markers + categorical_markers:
                if marker in key and marker in numeric_markers:
                    numeric_feature_columns.append(
                        tf.feature_column.numeric_column(key=key)
                    )
                elif marker in key and marker in categorical_markers:
                    hash_column = tf.feature_column.categorical_column_with_hash_bucket(
                        key=key,
                        hash_bucket_size=10000,
                        dtype=tf.int32
                    )
                    wide_categorical_columns.append(hash_column)
                    embedding_column = tf.feature_column.embedding_column(
                        categorical_column=hash_column,
                        dimension=10       # 10,000**0.25, per tf docs
                    )
                    deep_categorical_columns.append(embedding_column)

        if self.model_type == 'linear':
            classifier = tf.estimator.LinearClassifier(
                model_dir='{path}\linear models\{hash}'.format(path=path_to_tf_files, hash=self.save_hash),
                feature_columns=numeric_feature_columns + wide_categorical_columns,
                n_classes=max_number_of_wrestlers + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'deep':
            classifier = tf.estimator.DNNClassifier(
                model_dir='{path}\deep models\{hash}'.format(path=path_to_tf_files, hash=self.save_hash),
                feature_columns=numeric_feature_columns + deep_categorical_columns,
                hidden_units=self.layer_specs,
                n_classes=max_number_of_wrestlers + 1,  # labels must be strictly less than classes
            )
        elif self.model_type == 'hybrid':
            classifier = tf.estimator.DNNLinearCombinedClassifier(
                model_dir='{path}\hybrid models\{hash}'.format(path=path_to_tf_files, hash=self.save_hash),
                linear_feature_columns=numeric_feature_columns + wide_categorical_columns,
                dnn_feature_columns=numeric_feature_columns + deep_categorical_columns,
                dnn_hidden_units=self.layer_specs,
                n_classes=max_number_of_wrestlers + 1,  # labels must be strictly less than classes
            )

        # Train the Model.
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


def test():
    l_model = Model(batch_size=1, train_steps=1, model_type='linear', dataset_dict={'matches': 10}, name='Linear Test')
    d_model = Model(batch_size=1, train_steps=1, model_type='deep', dataset_dict={'matches': 10}, name='Deep Test')
    d_model = Model(batch_size=1, train_steps=1, model_type='hybrid', dataset_dict={'matches': 10}, name='Hybrid Test')
    pass


if __name__ == '__main__':
    test()
