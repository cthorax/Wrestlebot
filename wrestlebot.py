import tensorflow as tf
import numpy as np

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

class Model(object):
    def __init__(self):
        # called when created
        if SOMETHING:
            self.load_model()
        else:
            self.train_model()
        pass

    def input_to_model(self):
        # tf requires a function to put info into the model
        pass

    def load_model(self):
        # loads a saved model
        pass

    def save_model(self):
        # saves a model
        pass

    def train_model(self):
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