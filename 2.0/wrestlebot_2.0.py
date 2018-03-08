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
num2text_dict = {2: "wwe", 103: "nxt"}

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
    0: None,
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



mongo_address = os.getenv("mongo_address")

client = pymongo.MongoClient(mongo_address)
client.is_mongos  # blocks until connected to server
db = client["Wrastlin"]
collection = db["Matches 2.0"]

def parse(card, lastdate, lastpage):
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

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    try:
        for index_counter in range(1,lastpage).__reversed__():
            browser.get(index_url_template.format(index_counter, card))
            print("\npage {} beginning.\n".format(index_counter))

            for link_counter in range(2,12).__reversed__():
                date = browser.find_element_by_css_selector(index_date_template.format(link_counter)).text
                month, day, year = date.split()
                date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
                date = int(dt.datetime.strftime(date, "%Y%m%d"))
                if date <= lastdate:
                    print("date {} is less than {}, skipping.".format(date,lastdate))
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

                        matchtype = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 6)).text
                        matchtype = matchtype.lower()
                        if matchtype == "":
                            matchtype = "normal"

                        titles = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 7)).text
                        if titles == "":
                            titles = False
                        else:
                            titles = True

                        wrestlers = winners + losers

                        dict_entry = {
                            "date": date,
                            "winners": winners,
                            "wintype": wintype,
                            "wrestlers": wrestlers,
                            "raw_matchtype": matchtype,
                            "titles": titles,
                            "card": num2text_dict[card]
                            }
                        result = collection.insert_one(dict_entry)

                    except NoSuchElementException:
                        break

                browser.back()
    except NoSuchWindowException:
        print("selenium window crashed'")

    return


def get_training_data(entries):
    random_sample = collection.aggregate([
        {
            "$match": {
                "raw_matchtype": " "
            }
        },
        {
            "$sample": {
                "size": entries
            }
        }
    ])

    training_data = []
    valid_wintypes = list(wintype_antidict)
    valid_wintypes.sort()
    for item in valid_wintypes:
        entry = [item,0,0,0,0,0,0,0,0,0,0]
        training_data.append(entry)
    for match in random_sample:
        for name in match["wrestlers"]:
            match_history = get_db_wrestler(name, match["date"], 11)
            training_data.append(match_history)

    return training_data


def build_model(wrestlers):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    wrestler_array = np.asarray(wrestlers)
    training_array = wrestler_array[..., 1:]
    classifying_array = wrestler_array.transpose()[:1].transpose().flatten()

    model = SVC(probability=True)
    model.fit(training_array, classifying_array)

    scores = cross_val_score(model, training_array, classifying_array)
    print("Accuracy: {:0.2%} (+/- {:0.2%})".format(scores.mean(), scores.std() * 2))

    with open("model.pickle", "wb") as model_object:
        pickle.dump(model, model_object)

    return model


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


def last_date(card):
    try:
        lastdate = collection.find_one({"card": num2text_dict[card]}, sort=[('date', pymongo.DESCENDING)])['date']
    except TypeError:  # will fire off if there's no results, as there's no 'date' subscript to a None
        lastdate = 20140101
    return lastdate


def last_page(card):
    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    browser.get(index_url_template.format(1, card))
    lastpage = browser.find_element_by_css_selector(last_page_template)
    lastpage = lastpage.get_attribute('innerHTML')
    lastpage = int(int(lastpage.split("showing 1-10 of ")[1][:-1])/10)+1
    lastpage = min(lastpage, 400)
    return lastpage


def get_db_wrestler(name, date=21000101, matches=10):
    wrestler_exists = collection.find_one({"wrestlers": name})
    if wrestler_exists:
        temp_list = []
        wrestler_history = collection.find({
            "$and": [
                {"wrestlers": name},
                {"date": {"$lt": date}}
            ]
        }).limit(matches).sort("date", pymongo.DESCENDING)

        wrestler_list = list(wrestler_history)

        while len(wrestler_list) < matches:
            wrestler_list.append(
                {'winners': [], 'losers': [], 'date': 20140101, 'wintype': None}
            )

        for entry in wrestler_list:
            """
            date_as_date = dt.datetime.strptime(str(entry['date']), "%Y%m%d")
            if temp_list:
                days_since_last_match = previous_match - date_as_date
                days_since_last_match = days_since_last_match.days
                temp_list.append(days_since_last_match)
            previous_match = date_as_date
            """

            try:
                wintype = wintype_dict[entry['wintype']]
            except KeyError:
                print("new key value you didn't account for here.")

            if name not in entry['winners'] and wintype < 90:
                wintype = 50 + wintype

            temp_list.append(wintype)

        return temp_list
    else:
        print('no such wrestler as {}.\n'.format(name))
        return None


def main2():
    while 1:
        print("\n\n\n\n\n")
        print("1. Force Scrape\n2. Build Model\n\n0. Quit")
        menu_input = input()

        if menu_input == "1":
            for text_card in ['wwe', 'nxt']:
                card = text2num_dict[text_card]
                lastdate = last_date(card)
                lastpage = last_page(card)
                print("beginning {} scrape. last page is {}, last date was {}".format(text_card, lastpage, lastdate))
                matches = parse(card, lastdate, lastpage)

        if menu_input == "2":
            sample_size = int(input("how many samples? "))
            print("creating random sample of {} matches.".format(sample_size))
            training_data = get_training_data(sample_size)
            model = build_model(training_data)

        else:
            print('fuck off.')
            quit()

if __name__ == '__main__':
    main2()
