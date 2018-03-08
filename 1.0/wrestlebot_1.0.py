import pickle
import numpy as np

import os
import pymongo

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, NoSuchWindowException

import datetime as dt
import unicodedata

path_to_chromedriver = 'D:\Python\chromedriver.exe'
chrome_options = Options()
chrome_options.add_experimental_option('prefs', {'profile.managed_default_content_settings.cookies': 2,
                                                 'profile.managed_default_content_settings.images': 2,
                                                 'profile.managed_default_content_settings.javascript': 2,
                                                 'profile.managed_default_content_settings.plugins': 2,
                                                 'profile.managed_default_content_settings.popups': 2,
                                                 'profile.managed_default_content_settings.geolocation': 2,
                                                 'profile.managed_default_content_settings.notifications': 2,
                                                 'profile.managed_default_content_settings.media_stream': 2})

text2num_dict = {"wwe": 2, "nxt": 103}

index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='  # goes from 1 to whatever
event_link_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(3) > a"
last_page_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > div"

date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > table > tbody > tr:nth-child(1) > td:nth-child(1) > a"
index_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1) > a"
match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({})"
match_winner_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(2) > a:nth-child({})"
alternate_winner_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(2) > i > a:nth-child({})"
match_loser_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(4) > a:nth-child({})"
alternate_loser_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(4) > i > a:nth-child({})"

mongo_address = os.getenv("mongo_address")

client = pymongo.MongoClient(mongo_address)
client.is_mongos  # blocks until connected to server
db = client["Wrastlin"]
collection = db["Matches"]

def parse(card, lastdate, lastpage):

    match = []

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    try:
        for index_counter in range(1,lastpage).__reversed__():
            browser.get(index_url_template.format(index_counter, card))

            for link_counter in range(2,12).__reversed__():
                date = browser.find_element_by_css_selector(index_date_template.format(link_counter)).text
                month, day, year = date.split()
                date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
                if date <= lastdate:
                    continue
                else:
                    browser.find_element_by_css_selector(event_link_template.format(link_counter)).click()

                for match_counter in range(2,500):
                    try:
                        winners = []
                        for winner_counter in range(1,20):
                            try:
                                temp_winner = browser.find_element_by_css_selector(match_winner_template.format(
                                    match_counter, winner_counter)).text
                                temp_winner = unicodedata.normalize('NFD', temp_winner).encode(
                                    'ascii', 'ignore').decode()
                                temp_winner = temp_winner.lower()
                                winners.append(temp_winner)
                            except NoSuchElementException:
                                try:
                                    temp_winner = browser.find_element_by_css_selector(
                                        alternate_winner_template.format(match_counter,winner_counter)).text
                                    temp_winner = unicodedata.normalize('NFD', temp_winner).encode(
                                        'ascii', 'ignore').decode()
                                    temp_winner = temp_winner.lower()
                                    winners.append(temp_winner)
                                    break
                                except NoSuchElementException:
                                    continue

                        wintype = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 3)).text
                        wintype = wintype.lower()
                        if wintype == 'def. (pin)':
                            wintype = 2
                        elif wintype == 'def. (sub)':
                            wintype = 3
                        elif wintype == 'def. (dq)':
                            wintype = 4
                        elif wintype == 'def.':
                            wintype = 1
                        else:
                            wintype = 1

                        losers = []
                        for loser_counter in range(1,20):
                            try:
                                temp_loser = browser.find_element_by_css_selector(match_loser_template.format(
                                    match_counter, loser_counter)).text
                                temp_loser = unicodedata.normalize('NFD', temp_loser).encode(
                                    'ascii', 'ignore').decode()
                                temp_loser = temp_loser.lower()
                                losers.append(temp_loser)
                            except NoSuchElementException:
                                try:
                                    temp_loser = browser.find_element_by_css_selector(
                                        alternate_loser_template.format(match_counter, loser_counter)).text
                                    temp_loser = unicodedata.normalize('NFD', temp_loser).encode(
                                        'ascii', 'ignore').decode()
                                    temp_loser = temp_loser.lower()
                                    losers.append(temp_loser)
                                    break
                                except NoSuchElementException:
                                    continue

                        duration = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 5)).text

                        try:
                            m, s = duration.split(':')
                            duration = 60 * int(m) + int(s)
                        except ValueError:  # value error will happen if duration is empty or '<i></i>'
                            duration = -1

                        entry = [winners, wintype, losers, duration, date]
                        dict_entry = {
                            "date": date,
                            "winners": winners,
                            "wintype": wintype,
                            "losers": losers,
                            "duration": duration
                            }
                        match.append(entry)
                        result = collection.insert_one(dict_entry)

                    except NoSuchElementException:
                        break

                browser.back()
    except NoSuchWindowException:
        print("selenium window crashed, dumping existing match info as 'crash_matches.pickle'")
        with open("crash_matches.pickle", "wb") as raw_object:
            pickle.dump(match, raw_object)
        raise

    with open("raw_matches.pickle", "wb") as raw_object:
        pickle.dump(match, raw_object)

    return match


def sort_matches(matches):  # todo-kill this section, do all modeling via mongo
    # this bit takes the matches and sorts them into entries by wrestler

    max_matches = 10  # this may be changed
    wrestler_dict = dict()
    ignore_duration = True  #todo-fix ignore duration flag

    for entry in matches:
        if not entry[0]:  # entry without a winner
            continue

        match_length = entry[3]
        if not match_length or ignore_duration:
            match_length = -1

        for wrestler in entry[0]:
            if wrestler not in wrestler_dict:
                wrestler_dict[wrestler]=[]
                wrestler_dict[wrestler].append(wintype)
                wrestler_dict[wrestler].append(match_length)
            else:
                wrestler_dict[wrestler].append(wintype)
                wrestler_dict[wrestler].append(match_length)

        for wrestler in entry[2]:
            if wrestler not in wrestler_dict:
                wrestler_dict[wrestler]=[]
                wrestler_dict[wrestler].append(0 - wintype)
                wrestler_dict[wrestler].append(match_length)
            else:
                wrestler_dict[wrestler].append(0-wintype)
                wrestler_dict[wrestler].append(match_length)

    num_matches = 2*(max_matches+1)

    wrestler_array = np.asarray(num_matches*[0])  # empty array to be concatenated with each new wrestler array

    for wrestler in wrestler_dict:
        if len(wrestler_dict[wrestler]) > num_matches:
            temp_list = wrestler_dict[wrestler][0:num_matches]
        elif len(wrestler_dict[wrestler]) < num_matches:
            temp_list = wrestler_dict[wrestler] + (num_matches-len(wrestler_dict[wrestler])) * [0]

        temp_array = np.asarray(temp_list)
        wrestler_array = np.vstack((wrestler_array,temp_array))

    wrestler_array = wrestler_array[1:]  # strips off blank array
    return wrestler_array


def build_model(wrestlers):
    # this takes the lists of results for each wrestler and passes it through scikit-learn
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score

    training_array = wrestlers[..., 2:]
    classifying_array = wrestlers.transpose()[:1].transpose().flatten()

    model = linear_model.LogisticRegression(class_weight='balanced', C=0.001)
    model.fit(training_array, classifying_array)
    # looks at everything but the most recent match as the data to create the model and looks at just the wintype data
    # to classify everything

    scores = cross_val_score(model, training_array, classifying_array)
    print("Accuracy: {:0.2%} (+/- {:0.2%})".format(scores.mean(), scores.std() * 2))

    with open("model.pickle", "wb") as model_object:
        pickle.dump(model, model_object)

    return model


def predict(model, queries):
    for wrestler in queries:
        prediction = model.predict(wrestler)
        prediction = model.predict_proba(wrestler)


    return prediction


def test_model(model, event_dict):
    wintype_dict = {
        -4: "lose by disqualification",
        -3: "lose by submission",
        -2: "lose by pinfall",
        -1: "unspecified loss",
        0: "unknown / draw",
        1: "unspecified win",
        2: "win by pinfall",
        3: "win by submission",
        4: "win by disqualification",
    }

    '''
    template_list = [
        {"teams": ["",""],
            "matches" = [  # 
                [
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1
                ]  # 
            ],
            [  # 
                 [
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1,
                    , -1
                 ]  # 
            ]
        ],
    ]
    '''

    for match in event_dict['matches']:
        match_result = []
        for team in match:
            team_result = []
            for wrestler in team:
                temp_array = np.asarray([[wrestler]])
                reshaped_array = temp_array
                temp_result = predict(model, reshaped_array)
                team_result.append(temp_result)

            team_array = np.asarray(team_result).mean(axis=0)
            win = sum(team_array[0][4:])

            match_result.append(win)

        print("match results:\n")
        norm = sum(match_result)

        for wrestler in range(len(event_dict["teams"][0])):
            print("{}:     {:.3%}".format(event_dict["teams"][0][wrestler], match_result[wrestler]/norm))

        winner = match_result.index(max(match_result))
        print("predicted winner is {}.\n".format(event_dict["teams"][0][winner]))

    return


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


def load_raw():
    try:
        with open("raw_matches.pickle", "rb") as matches_object:
            matches = pickle.load(matches_object)
    except FileNotFoundError:
        print("no raw matches file located.")
        quit()

    return matches


def last_date():
    try:
        lastdate = collection.find_one(sort=[('date', pymongo.DESCENDING)])['date']
    except TypeError:  # will fire off if there's no results, as there's no 'date' subscript to a None
        lastdate = dt.datetime(1901, 1, 1)
    return lastdate


def last_page(card):
    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    browser.get(index_url_template.format(1, card))
    lastpage = browser.find_element_by_css_selector(last_page_template)
    lastpage = lastpage.get_attribute('innerHTML')
    lastpage = int(int(lastpage.split("showing 1-10 of ")[1][:-1])/10)+1
    return lastpage


def get_db_wrestler(name):
    temp_list = []
    wrestler = collection.find({"$or": [
        {"winners": name}, {"losers": name}]}).limit(10).sort("date", pymongo.DESCENDING)

    if wrestler.count(True):  # needs True flag in order to accurately count with limits and skips
        previous_match = None
        wrestler_list = list(wrestler)

        while len(wrestler_list) < 10:
            wrestler_list.append(
                {'duration': -1, 'losers': [], 'date': dt.datetime(1901, 1, 1, 0, 0), 'wintype': 0}
            )

        for entry in wrestler_list:
            wintype = entry['wintype']
            if name in entry['losers']:
                wintype = -wintype

            temp_list.append(wintype)

            duration = entry['duration']
            duration = -1
            temp_list.append(duration)

            if previous_match:
                days_since_last_match = previous_match - entry['date']
            else:
                days_since_last_match = 0
            previous_match = entry['date']
            # templist.append(days_since_last_match)



        return temp_list

    else:
        print('no such wrestler as {}.\n'.format(name))
        return None


def single_predict():
    print("Enter wrestlers one team at a time. Leave prompt blank to finish team, leave prompt blank to close teams.")
    teams = []
    team_names = []
    while 1:
        print("Begin Team {}".format(len(teams)+1))
        temp_team = []
        temp_team_names = []
        while 1:
            name_input = input("Enter wrestler name: ").lower()
            if name_input:
                temp_wrestler = get_db_wrestler(name_input)
                if temp_wrestler:
                    temp_team_names.append(name_input)
                    temp_team.append(temp_wrestler)
                    print("{} added to team. {} members so far.".format(name_input, len(temp_team)))

            else:
                if temp_team:
                    print("closing team.\n")
                    teams.append(temp_team)
                    team_names.append(temp_team_names)
                    break
                else:
                    print("all teams entered, finishing setup.")
                    break
        if not temp_team:
            break

    event_dict = {"teams": [team_names], "matches": [teams]}
    test_model(load_model(), event_dict)


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


    for match in evaluated_list:
        teams = []
        team_names = []
        for team in match:
            temp_team = []
            temp_team_names = []
            for wrestler in team:
                temp_wrestler = get_db_wrestler(wrestler)
                if temp_wrestler:
                    temp_team_names.append(wrestler)
                    temp_team.append(temp_wrestler)
                else:
                    break

            teams.append(temp_team)
            team_names.append(temp_team_names)

        event_dict = {"teams": [team_names], "matches": [teams]}
        test_model(load_model(), event_dict)


def main2():
    while 1:
        print("\n\n\n\n\n")
        print("1. Test Model\n2. Test Scrape\n3. Parse Raw Matches\n4. Force Scrape\n5. Single Predict\n6. List Predict\n\n0. Quit")
        menu_input = input()

        if menu_input == "1":
            test_model(load_model())

        elif menu_input == "2":
            lastdate = dt.datetime(1901, 1, 1)
            lastpage = 2

            print("beginning scrape.")
            matches = parse(lastdate, lastpage)
            print("scrape finished.")
            wrestlers = sort_matches(matches)
            print("matches processed.")
            model = build_model(wrestlers)
            print("model built.")

        elif menu_input == "3":
            matches = load_raw()
            print("raw matches loaded.")
            wrestlers = sort_matches(matches)
            print("matches processed.")
            model = build_model(wrestlers)
            print("model built.")

        elif menu_input == "4":

            for text_card in ['wwe', 'nxt']:
                card = text2num_dict[text_card]
                lastdate = last_date()
                lastpage = last_page(card)

                print("beginning {} scrape.".format(text_card))
                matches = parse(card, lastdate, lastpage)
            """
            print("scrape finished.")
            wrestlers = sort_matches(matches)
            print("matches processed.")
            model = build_model(wrestlers)
            print("model built.")

            """

        elif menu_input == "5":
            single_predict()

        elif menu_input == "6":
            list_predict()

        else:
            print('fuck off.')
            quit()

if __name__ == '__main__':
    main2()
