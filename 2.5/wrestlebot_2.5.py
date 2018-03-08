import pickle
import numpy as np

import os
import pymongo

from math import fsum  # fsum is better for floats
from statistics import mean

mongo_address = os.getenv("mongo_address")

client = pymongo.MongoClient(mongo_address)
client.is_mongos  # blocks until connected to server
db = client["Wrastlin"]
collection = db["Matches 2.0"]

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


def load_model():
    model_file = "model.pickle"
    try:
        with open(model_file,"rb") as model_object:
            model = pickle.load(model_object)
        print("model file loaded.")
    except FileNotFoundError:
        print("no model file located.")
        quit()

    return model


class Wrestler(object):
    def __init__(self, name):
        self.name = name.lower()
        self.update()

    def update(self):
        # this will update all relevant wrestler stats when called
        self.get_history()
        self.get_predictions()

    def get_history(self):
        # queries DB for history, deletes instance if wrestler is not found.
        if collection.find_one({"wrestlers": self.name}):  # checks the DB for an instance of the wrestler, and fails if none exists
            raw_history = list(collection.find({"wrestlers": self.name}).limit(10).sort("date", pymongo.DESCENDING))
        else:
            print("wrestler does not exist.")
            del self

        partial_history = []
        for entry in raw_history:
            try:
                wintype = wintype_dict[entry['wintype']]
            except KeyError:
                print("new key value you didn't account for here.")

            if self.name not in entry['winners'] and wintype < 90:
                wintype = 50 + wintype

            partial_history.append(wintype)

        while len(partial_history) < 10:
            partial_history.append(0)

        self.history = partial_history

    def get_predictions(self):
        history_array = np.asarray([self.history])
        prediction = loaded_model.predict_proba(history_array)

        win_probability = []
        lose_probability = []
        draw_probability = []

        for i, j in enumerate(sorted(wintype_antidict.keys())):
            if wintype_antidict[j][:3] == "win":
                win_probability.append(prediction[..., i][0])
            elif wintype_antidict[j][:4] == "draw":
                draw_probability.append(prediction[..., i][0])
            elif wintype_antidict[j][:4] == "loss":
                lose_probability.append(prediction[..., i][0])

        win_probability = fsum(win_probability)
        lose_probability = fsum(lose_probability)
        draw_probability = fsum(draw_probability)

        self.prediction = prediction
        self.wld_tuple = (win_probability, lose_probability, draw_probability)


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
        win_probability = []
        lose_probability = []
        draw_probability = []
        prediction = np.zeros((1,24))
        team_name = ""

        for wrestler in self.members:
            win_probability.append(wrestler.wld_tuple[0])
            lose_probability.append(wrestler.wld_tuple[1])
            draw_probability.append(wrestler.wld_tuple[2])
            prediction = np.vstack((prediction, wrestler.prediction))
            team_name = ", ".join([team_name, wrestler.name])

        win_probability = mean(win_probability)
        lose_probability = mean(lose_probability)
        draw_probability = mean(draw_probability)

        np.delete(prediction, 0, 0)  # removes blank row from top.
        prediction = list(prediction.mean(axis=0))

        team_name = team_name[2:]  # removes extra ', ' from front
        team_name = ', & '.join(team_name.rsplit(', ',1))
        if team_name.count(',') == 1:  # oxford comma fixer
            team_name = team_name.replace(',', '')


        self.wld_tuple = (win_probability, lose_probability, draw_probability)
        self.team_name = team_name
        self.prediction = prediction


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
        total_losses = 1
        total_draws = 1
        match_name = ""

        for team in self.teams:
            total_losses *= team.wld_tuple[1]
            total_draws *= team.wld_tuple[2]
            match_name = " VS ".join([match_name, team.team_name])

        prediction_list = []
        for team in self.teams:
            result = total_losses * team.wld_tuple[0] / team.wld_tuple[1]
            prediction_list.append(result)
        prediction_list.append(total_draws)

        prediction = max(prediction_list)
        if prediction == prediction_list[-1]:  # [-1] will always be draw
            draw_type = np.zeros((1, 6))
            for team in self.teams:
                draw_type = np.vstack((draw_type, np.asarray(team.prediction[17:-1])))

            draw_type = np.delete(draw_type, 0, 0)  # deletes blank row
            draw_type = list(np.prod(draw_type, axis=0))  # vertical product of all rows
            index = draw_type.index(max(draw_type))
            draw_type = wintype_antidict[90+index]
            prediction = "{}".format(draw_type)
        else:
            index = prediction_list.index(prediction)
            result_list = sorted(wintype_antidict.keys())[1:9]
            target = max(self.teams[index].prediction[1:9])
            for i, j in enumerate(result_list):
                if self.teams[index].prediction[i] == target:
                    wintype = j
                    break
            prediction = "{}, {}".format(self.teams[index].team_name, wintype_antidict[wintype])

        self.match_name = match_name[4:]  # removes extra ' VS ' from front
        self.prediction = prediction
        self.verbose_prediction = prediction_list


def list_predict():
    print("enter lists for matches. input should be a triple-embedded list: Matches, Teams, Wrestlers. Don't put spaces after commas.")
    from ast import literal_eval

    evaluated_list = None
    while not evaluated_list:
        list_input = input("Enter list: ").lower()
        evaluated_list = literal_eval(list_input)
        check_type = type(evaluated_list)
        if not check_type == list:  #is it actually a list
            evaluated_list = None
        if not evaluated_list[0][0][0]:  #is it triple nested
            evaluated_list = None

    event = []
    for match in evaluated_list:
        temp_match = []
        for team in match:
            temp_team = []
            for wrestler in team:
                temp_wrestler = Wrestler(name=wrestler)
                if temp_wrestler:
                    temp_team.append(temp_wrestler)
                else:
                    break

            temp_team = Team(members=temp_team)
            temp_match.append(temp_team)

        temp_match = Match(teams=temp_match)
        event.append(temp_match)

    for match in event:
        print("{}:\n\t{}\n".format(match.match_name, match.prediction))

def main():
    global loaded_model
    loaded_model = load_model()

    list_predict()

if __name__ == '__main__':
    main()