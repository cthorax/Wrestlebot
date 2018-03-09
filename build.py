
import os
import pymongo

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions

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
collection = db["Matches 3"]

def parse(card, lastdate, lastpage):
    index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='  # goes from 1 to whatever
    event_link_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(3) > a"

    index_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1) > a"
    match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({})"
    match_winner_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(2) > a:nth-child({})"
    alternate_winner_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(2) > i > a:nth-child({})"
    match_loser_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(4) > a:nth-child({})"
    alternate_loser_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(4) > i > a:nth-child({})"
    duration_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(5)"
    match_number_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1)"

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    try:
        browser.get(index_url_template.format(1, card))
        date = browser.find_element_by_css_selector(index_date_template.format(2)).text
        month, day, year = date.split()
        date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
        date = int(dt.datetime.strftime(date, "%Y%m%d"))
        if date <= lastdate:
            print("all records up to date.\n\n\n\n")
            return True

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
                        match_number = browser.find_element_by_css_selector(match_number_template.format(match_counter)).text

                        winners = []
                        for winner_counter in range(1,20):
                            try:
                                temp_winner = browser.find_element_by_css_selector(match_winner_template.format(
                                    match_counter, winner_counter)).text
                                temp_winner = unicodedata.normalize('NFD', temp_winner).encode(
                                    'ascii', 'ignore').decode()
                                temp_winner = temp_winner.lower()
                                winners.append(temp_winner)
                            except selenium.common.exceptions.NoSuchElementException:
                                try:
                                    temp_winner = browser.find_element_by_css_selector(
                                        alternate_winner_template.format(match_counter,winner_counter)).text
                                    temp_winner = unicodedata.normalize('NFD', temp_winner).encode(
                                        'ascii', 'ignore').decode()
                                    temp_winner = temp_winner.lower()
                                    winners.append(temp_winner)
                                    break
                                except selenium.common.exceptions.NoSuchElementException:
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
                            except selenium.common.exceptions.NoSuchElementException:
                                try:
                                    temp_loser = browser.find_element_by_css_selector(
                                        alternate_loser_template.format(match_counter, loser_counter)).text
                                    temp_loser = unicodedata.normalize('NFD', temp_loser).encode(
                                        'ascii', 'ignore').decode()
                                    temp_loser = temp_loser.lower()
                                    losers.append(temp_loser)
                                    break
                                except selenium.common.exceptions.NoSuchElementException:
                                    continue

                        duration = browser.find_element_by_css_selector(duration_template.format(match_counter)).text
                        duration = duration.split(":")
                        if len(duration) == 2:  # split on a string without the seperator returns a list containing the string
                            duration = float(duration[0]) + float(duration[1])/60

                        matchtype = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 6)).text
                        matchtype = matchtype.lower()

                        titles = browser.find_element_by_css_selector(
                            match_text_template.format(match_counter, 7)).text

                        wrestlers = winners + losers

                        id = "".join([str(date), match_number, "_", str(card)])

                        dict_entry = {
                            "_id": id,
                            "date": date,
                            "duration": duration,
                            "winners": winners,
                            "wintype": wintype,
                            "wrestlers": wrestlers,
                            "matchtype": matchtype,
                            "titles": titles,
                            "card": num2text_dict[card]
                            }

                        try:
                            result = collection.insert_one(dict_entry)
                        except pymongo.errors.DuplicateKeyError:
                            break

                    except selenium.common.exceptions.NoSuchElementException:
                        break

                browser.back()
    except selenium.common.exceptions.WebDriverException:
        print("selenium window crashed'")

    return True


def last_date(card):
    try:
        lastdate = collection.find_one({"card": num2text_dict[card]}, sort=[('date', pymongo.DESCENDING)])['date']
    except TypeError:  # will fire off if there's no results, as there's no 'date' subscript to a None
        lastdate = 20140101
    return lastdate


def last_page(card):
    index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='
    last_page_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > div"

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    browser.get(index_url_template.format(1, card))
    lastpage = browser.find_element_by_css_selector(last_page_template)
    lastpage = lastpage.get_attribute('innerHTML')
    lastpage = int(int(lastpage.split("showing 1-10 of ")[1][:-1])/10)+1
    lastpage = min(lastpage, 400)
    return lastpage


def main():
    for text_card in ['wwe', 'nxt']:
        matches = False
        while not matches:
            card = text2num_dict[text_card]
            lastdate = last_date(card)
            lastpage = last_page(card)
            print("beginning {} scrape. last page is {}, last date was {}".format(text_card, lastpage, lastdate))
            matches = parse(card, lastdate, lastpage)


if __name__ == '__main__':
    main()
