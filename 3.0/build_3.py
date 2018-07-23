
import os
import pymongo

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions

import datetime as dt

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
match_collection = db["Matches 3"]
wrestler_collection = db["Wrestlers"]

index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='  # goes from 1 to whatever
single_match_template = 'http://www.profightdb.com/cards/{}.html'   #format is 'card/event-name'

event_link_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(3) > a"

index_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1) > a"
match_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > table > tbody > tr:nth-child(1) > td:nth-child(1) > a"

match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({})"
duration_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(5)"

other_match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({}) > a:nth-child({})"
alternate_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > i > td:nth-child({}) > a:nth-child({})"



def renamethislater(match_counter, browser, type):

    if type == "winner":
        type_number = 2
    elif type == "loser":
        type_number = 4

    temp_names = []
    temp_urls = []
    for counter in range(1, 20):
        try:
            url = browser.find_element_by_css_selector(other_match_text_template.format(
                match_counter, type_number, counter)).get_attribute("href")[36:-5]
            name = browser.find_element_by_css_selector(other_match_text_template.format(
                match_counter, type_number, counter)).text
            name = clean_text(name)
            temp_names.append(name)
            temp_urls.append(url)
        except selenium.common.exceptions.NoSuchElementException:
            try:
                url = browser.find_element_by_css_selector(alternate_text_template.format(
                    match_counter, type_number, counter)).get_attribute("href")[36:-5]
                name = browser.find_element_by_css_selector(
                    alternate_text_template.format(match_counter, type_number, counter)).text
                name = clean_text(name)
                temp_names.append(name)
                temp_urls.append(url)
                break
            except selenium.common.exceptions.NoSuchElementException:
                break

    return temp_names, temp_urls


def clean_text(text):
    import unicodedata

    temp_string = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode()
    temp_string = temp_string.lower()
    temp_string = temp_string.replace("'", "")
    temp_string = temp_string.replace('"', "")
    temp_string = temp_string.replace('.', "")
    return temp_string


def full_parse(card, lastdate, lastpage, update=False):

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    try:
        browser.get(index_url_template.format(1, card))
        date = browser.find_element_by_css_selector(index_date_template.format(2)).text
        month, day, year = date.split()
        date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
        date = int(dt.datetime.strftime(date, "%Y%m%d"))
        try:
            date <= lastdate
        except TypeError:
            lastdate = int(lastdate)    # these all should be int already but some probably slipped through the db change
        if date <= lastdate and not update:
            print("all records up to date.")
        else:
            if update:
                page_range = range(1,lastpage)
                link_range = range(2,12)
            else:
                page_range = range(1, lastpage).__reversed__()
                link_range = range(2,12).__reversed__()

            for index_counter in page_range:
                browser.get(index_url_template.format(index_counter, card))
                print("\npage {} beginning.\n".format(index_counter))

                for link_counter in link_range:
                    date = browser.find_element_by_css_selector(index_date_template.format(link_counter)).text
                    month, day, year = date.split()
                    date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
                    date = int(dt.datetime.strftime(date, "%Y%m%d"))
                    if date <= lastdate and not update:
                        print("date {} is less than {}, skipping.".format(date,lastdate))
                        continue
                    else:
                        browser.find_element_by_css_selector(event_link_template.format(link_counter)).click()

                    single_parse(browser=browser, date=date, card=card, update=update)

    except selenium.common.exceptions.WebDriverException:
        try:
            browser.close()
        except selenium.common.exceptions.WebDriverException:
            pass

        print("selenium window crashed")

    try:
        browser.close()
    except selenium.common.exceptions.WebDriverException:
        pass

    return True


def single_parse(browser, date, card, update=False):
    url = browser.current_url[36:-5]
    for match_counter in range(2, 500):
        try:
            match_number = match_counter - 1
            wrestlers = []
            url_dict = {}

            winners, urls = renamethislater(match_counter, browser, type='winner')
            wrestlers.extend(winners)
            for key, value in zip(winners, urls):
                url_dict[key] = value

            wintype = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 3)).text
            wintype = wintype.lower()
            if not wintype:
                wintype = 'def.'  # only one blank wintype found so far but it was a huge PITA, so.

            losers, urls = renamethislater(match_counter, browser, type='loser')
            wrestlers.extend(losers)
            for key, value in zip(losers, urls):
                url_dict[key] = value

            teams = []
            raw_teams = browser.find_element_by_css_selector(match_text_template.format(match_counter, 4)).text
            if '&' in raw_teams or len(winners) > 1:
                teams.append(winners)
                raw_teams = clean_text(raw_teams)

                raw_teams = raw_teams.split(',')
                for team in raw_teams:
                    members = team.split(' & ')
                    temp_team = []
                    for wrestler in members:
                        temp_team.append(wrestler)
                    teams.append(temp_team)

            else:
                for wrestler in wrestlers:
                    teams.append(wrestler)

            duration = browser.find_element_by_css_selector(duration_template.format(match_counter)).text
            duration = duration.split(":")
            if len(duration) == 2:  # split on a string without the seperator returns a list containing the string
                duration = int(duration[0]) * 60 + int(duration[1])
            else:
                duration = 0

            matchtype = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 6)).text
            if matchtype == ' ':
                matchtype = 'normal'
            else:
                matchtype = clean_text(matchtype)

            titles = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 7)).text
            if titles:
                titles = clean_text(titles)
            else:
                titles = 'none'

            id = "".join([str(date), "-", str(match_number), "_", str(card), "_", url])

            dict_entry = {
                "_id": id,
                "date": date,
                "duration": duration,
                "winners": winners,
                "wintype": wintype,
                "wrestlers": wrestlers,
                "teams": teams,
                "matchtype": matchtype,
                "titles": titles,
                "card": num2text_dict[card]
            }

            if update:
                try:
                    result = match_collection.update_many({'_id': id}, {'$set': dict_entry}, upsert=True)
                    # print('matched entries: {}\nupdated entries: {}'.format(result.matched_count, result.modified_count))
                except OverflowError as e:
                    print(e)
                    for key, value in dict_entry:
                        print("{}: {}, type {}".format(key, value, type(value)))
                    pass
            else:
                try:
                    result = match_collection.insert_one(dict_entry)
                except pymongo.errors.DuplicateKeyError as e:
                    print('\nlook at this real quick:\n{}\n'.format(e))
                    pass

            for name, wrestler_url in url_dict.items():
                result = wrestler_collection.update_one({'_id': wrestler_url}, {'$addToSet': {'name': name}}, upsert=True)
                pass

        except selenium.common.exceptions.NoSuchElementException:
            print("event \'{}\' done.".format(url))
            break

    browser.back()
    return True


def last_date(card):
    try:
        lastdate = match_collection.find_one({"card": num2text_dict[card]}, sort=[('date', pymongo.DESCENDING)])['date']
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
    lastpage = min(lastpage, 200)
    browser.close()
    return lastpage


def main():
    print('\n\n\n\n\n')
    for text_card in ['wwe', 'nxt']:
        card = text2num_dict[text_card]
        lastdate = last_date(card)
        lastpage = last_page(card)
        print("beginning {} scrape. last page is {}, last date was {}".format(text_card, lastpage, lastdate))
        matches = full_parse(card, lastdate, lastpage, update=True)   # update=True is for testing / updating


if __name__ == '__main__':
    main()
