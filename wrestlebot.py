import tensorflow as tf
import numpy as np
import pymongo

from io import TextIOBase as file_type # whatever the type of a file winds up being

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
    None: 0
}
wintype_antidict = {
    0: 'none',
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
    6: 'win count-out',
    7: 'win via ko',
    8: 'win via tko',
    90: 'draw',
    91: 'draw, no contest',
    92: 'draw, double count-out',
    93: 'draw, time out',
    94: "draw, double disqualification",
    95: "draw, double pin"
}
wintype_list = []
for position, item in enumerate(sorted(wintype_antidict.keys())):
    # this is dynamic in case i eventually wind up needing to add new wintypes. this all may be obviated by a new storage scheme.
    wintype_list[position] = item

class Wrestler(object, name):
    def __init__(self):
        assert name
        self.name = self.name.lower()
        self.update()

    def update(self):
        # this will update all relevant wrestler stats when called
        if match_collection.find_one({"wrestlers": self.name}):  # checks the DB for an instance of the wrestler, and fails if none exists
            raw_history = list(match_collection.find({"wrestlers": self.name}).limit(10).sort("date", pymongo.DESCENDING))
        else:
            print("wrestler does not exist.")
            del self

        get_history(raw_history)


        self.get_history(cursor)

    def get_history(self, history):
        # queries DB for history, deletes instance if wrestler is not found.

        partial_history = []
        for entry in history:
            try:
                wintype = wintype_dict[entry['wintype']]
            except KeyError:
                print("new key value you didn't account for here.")
                pass

            if self.name not in entry['winners'] and wintype < 90:
                wintype = 50 + wintype

            partial_history.append(wintype)

        while len(partial_history) < 10:
            partial_history.append(0)

        self.history = partial_history


class Team(object):
    def __init__(self, members):
        if isinstance(members, list):
            self.members = members
        else:
            self.members = [members]
        self.update()

    def add_members(self, new_members):
        if isinstance(new_members, list):
            self.members = self.members.extend(new_members)
        else:
            self.members = self.members.append(new_members)
        self.update()

    def update(self):
        # this will update all relevant team stats when called.
        team_name = ""

        for wrestler in self.members:
            team_name = ", ".join([team_name, wrestler.name])

        team_name = team_name[2:]  # removes extra ', ' from front
        team_name = ', & '.join(team_name.rsplit(', ',1))
        if team_name.count(',') == 1:  # oxford comma fixer
            team_name = team_name.replace(',', '')

        self.team_name = team_name


class Match(object):
    def __init__(self, teams):
        self.teams = teams
        self.update()

    def add_teams(self, new_teams):
        if isinstance(new_teams, list):
            self.members = self.members.extend(new_teams)
        else:
            self.members = self.members.append(new_teams)
        self.update()

    def update(self):
        #this will update all relevant Match stats when called.
        match_name = ""

        for team in self.teams:
            match_name = " VS ".join([match_name, team.team_name])

        self.match_name = match_name[4:]  # removes extra ' VS ' from front


class Dataset(object):
    def __init__(self):
        pass


class Model(object, file_or_dataset):
    def __init__(self):
        # called when created
        if isinstance(file_or_dataset, file_type):
            self.load_model(file_or_dataset)
        else:
            self.train_model(file_or_dataset)
        pass

    def input_to_model(self):
        # tf requires a function to put info into the model
        pass

    def load_model(self, load_file):
        # loads a saved model
        pass

    def save_model(self):
        # saves a model
        pass

    def train_model(self, dataset):
        # calls input_to_model to train a new model

        pass

    def make_prediction(self):
        # calls input_to_model to make a prediction
        pass

    def assess_model(self):
        # outputs the statistical efficacy of the model
        pass



def main():
    # the main goddamn program, duh
    pass

if __name__ == '__main__':
    main()