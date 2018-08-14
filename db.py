import records
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


team_start_marker = ''
team_end_marker = 'T'
wrestler_start_marker = 'x'
wrestler_end_marker = 'y'

def init_table(table_dict, db):
    """
    drops and creates a table based on a passed dict into the specified db

    :param table_dict:
    dict format below. at least one table_param entry required, no foreign_keys required
    {
        'table_name': string,
        'table_params': [
            {
                'key_name': string required,
                'key_type':  'NULL' or 'INTEGER' or 'REAL' or 'TEXT' or 'BLOB' required,
                'primary_key_flag': True or False,
                'unique_flag': True or False,
                'not_null_flag': True or False
            }
        ]
     }

    :param db:
    active records db instance

    :return:
    none
    """

    drop_template = 'DROP TABLE IF EXISTS {table_name};'
    create_template = 'CREATE TABLE {table_name} (\n{table_params}\n);'

    drop_query = drop_template.format(**table_dict)
    db.query(drop_query)

    param_queries = ''
    for param in table_dict['table_params']:

        if param.get('primary_key_flag', None):
            primary_key_flag = ' PRIMARY KEY'
            unique_flag = ''
            not_null_flag = ''
        else:
            primary_key_flag = ''
            if param.get('unique_flag', None):
                unique_flag = ' UNIQUE'
            else:
                unique_flag = ''
            if param.get('not_null_flag', None):
                not_null_flag = ' NOT NULL'
            else:
                not_null_flag = ''

        param_query = '{key_name} {key_type}{primary_key_flag}{unique_flag}{not_null_flag},\n'.format(
            key_name=param.get('key_name'),
            key_type=param.get('key_type'),
            primary_key_flag=primary_key_flag,
            unique_flag=unique_flag,
            not_null_flag=not_null_flag
        )

        param_queries += param_query

    param_queries = param_queries[:-2]  # strips trailing ',\n'

    create_query = create_template.format(table_name=table_dict.get('table_name'), table_params=param_queries)
    db.query(create_query)


def init_virtual_table(table_dict, db):
    """
    drops and creates a table based on a passed dict into the specified db

    :param table_dict:
    dict format below. at least one table_param entry required, no foreign_keys required
    {
        'table_name': string,
        'table_params': [
            {
                'key_name': string required
            }
        ]
     }

    :param db:
    active records db instance

    :return:
    none
    """

    drop_template = 'DROP TABLE IF EXISTS {table_name};'
    create_template = 'CREATE VIRTUAL TABLE {table_name} USING FTS5 (\n{table_params}\n);'

    drop_query = drop_template.format(**table_dict)
    db.query(drop_query)

    param_queries = ''
    for param in table_dict['table_params']:

        param_query = '{key_name}, '.format(key_name=param.get('key_name'))
        param_queries += param_query

    param_queries = param_queries[:-2]  # strips trailing ', '

    create_query = create_template.format(table_name=table_dict.get('table_name'), table_params=param_queries)
    db.query(create_query)


def full_parse(card_dict, lastdate, lastpage, browser, force_update=False):
    index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='  # goes from 1 to whatever
    event_link_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(3) > a"
    index_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1) > a"

    try:
        browser.get(index_url_template.format(1, card_dict['number']))
        date = browser.find_element_by_css_selector(index_date_template.format(2)).text
        month, day, year = date.split()
        date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
        date = int(dt.datetime.strftime(date, "%Y%m%d"))
        if date > lastdate or force_update:
            if force_update:
                page_range = range(1, lastpage)
            else:
                page_range = range(1, lastpage).__reversed__()

            for index_counter in page_range:
                if force_update:
                    link_range = range(2, 12)
                else:
                    link_range = range(2, 12).__reversed__()

                browser.get(index_url_template.format(index_counter, card_dict['number']))
                print("\npage {} beginning.\n".format(index_counter))

                for link_counter in link_range:
                    date = browser.find_element_by_css_selector(index_date_template.format(link_counter)).text
                    month, day, year = date.split()
                    date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
                    date = int(dt.datetime.strftime(date, "%Y%m%d"))
                    if date <= lastdate and not force_update:
                        print("date {} is less than {}, skipping.".format(date,lastdate))
                        continue
                    else:
                        browser.find_element_by_css_selector(event_link_template.format(link_counter)).click()

                    match_dict_list, wrestler_update_list = page_parse(browser=browser, date=date, card_dict=card_dict)
                    update_wrestler_table(wrestler_url_list=wrestler_update_list, db=db, browser=browser)
                    update_match_table(match_dict_list=match_dict_list, db=db)
        else:
            print("all records up to date.")

    except selenium.common.exceptions.WebDriverException:
        print("selenium window crashed")


def page_parse(browser, date, card_dict):
    match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({})"
    duration_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(5)"

    url = browser.current_url[36:-5]
    match_dict_list = []
    all_wrestler_urls = set()
    for match_counter in range(2, 500):
        try:
            match_number = match_counter - 1
            wrestler_urls = []

            winner_urls = match_parse(match_counter, browser, type='winner', db=db)
            wrestler_urls.extend(winner_urls)

            wintype = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 3)).text
            wintype = wintype.lower()
            if not wintype:
                wintype = 'def.'  # only one blank wintype found so far but it was a huge PITA, so.

            loser_urls = match_parse(match_counter, browser, type='loser', db=db)
            wrestler_urls.extend(loser_urls)

            teams = []
            raw_teams = browser.find_element_by_css_selector(match_text_template.format(match_counter, 4)).get_attribute('innerHTML')
            if '&amp;' in raw_teams and ',' in raw_teams:   # multiple teams, at least one team bigger than one person.
                teams.append(winner_urls)
                raw_teams = raw_teams.split(',')
                for team in raw_teams:
                    members = team.split(' &amp; ')
                    temp_team = []
                    for wrestler in members:
                        wrestler = wrestler.split('>')
                        temp_wrestler = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                        for split in wrestler:
                            if split.find('<a') == -1:
                                continue
                            elif split.find('<a') < len(temp_wrestler):
                                temp_wrestler = split

                        wrestler = temp_wrestler
                        wrestler = wrestler.split('/')[-1][:-6]

                        temp_team.append(wrestler)
                    teams.append(temp_team)

            elif '&amp;' in raw_teams and ',' not in raw_teams:     # only one losing team, with more than one person.
                teams.append(winner_urls)
                teams.append(loser_urls)

            else:   # multiple teams, all with one person, or single loser not in a team.
                teams.append(winner_urls)
                for wrestler in loser_urls:
                    teams.append([wrestler])

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
                matchtype = clean_text(matchtype).strip()

            titles = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 7)).text
            if titles:
                titles = clean_text(titles).strip()
            else:
                titles = 'none'

            match_id = "".join([str(date), "-", str(match_number), "_", str(card_dict['number']), "_", url])

            match_dict = {
                "match_id": match_id,
                "date": date,
                "duration": duration,
                "winners": winner_urls,
                "wintype": wintype,
                "wrestlers": wrestler_urls,
                "teams": teams,
                "matchtype": matchtype,
                "titles": titles,
                "card": card_dict['name']
            }

            match_dict_list.append(match_dict)
            all_wrestler_urls.update(wrestler_urls)

        except selenium.common.exceptions.NoSuchElementException:
            print("event \'{}\' done.".format(url))
            break

    browser.back()
    return match_dict_list, list(all_wrestler_urls)


def update_match_table(match_dict_list, db):
    """

    :param match_dict_list:
    a list where each element is a dict in the following form
    {
        "match_id": string,
        "date": integer,
        "duration": integer,
        "winners": list of wrestler_urls (strings),
        "wintype": string,
        "wrestlers": list of wrestler_urls (strings),
        "teams": list of list of wrestler_urls (strings),
        "matchtype": string,
        "titles": string,
        "card": string
    }

    :param db:
    active records db instance

    :return:
    none
    """
    match_insert_template = "INSERT INTO match_table (match_id, date, duration, wintype, titles, matchtype, card) VALUES ('{match_id}', {date}, {duration}, {wintype}, {titles}, {matchtype}, '{card}')"
    team_insert_template = "INSERT INTO team_table (match_id, competitors) VALUES ('{match_id}', '{competitors}')"

    for match_dict in match_dict_list:
        match_insert_dict = {
            'match_id': match_dict['match_id'],
            'date': match_dict['date'],
            'duration': match_dict['duration'],
            'wintype': -1,
            'titles': -1,
            'matchtype': -1,
            'card': match_dict['card']
        }

        wintype_query = db.query("SELECT id FROM wintype_table WHERE wintype = '{}'".format(match_dict["wintype"])).as_dict()
        if wintype_query:
            match_insert_dict['wintype'] = wintype_query[0].get('id')
        else:
            db.query("INSERT INTO wintype_table (wintype) VALUES ('{}')".format(match_dict["wintype"]))
            wintype_query = db.query("SELECT id FROM wintype_table WHERE wintype = '{}'".format(match_dict["wintype"])).as_dict()
            match_insert_dict['wintype'] = wintype_query[0].get('id')

        title_query = db.query("SELECT id FROM title_table WHERE title = '{}'".format(match_dict["titles"])).as_dict()
        if title_query:
            match_insert_dict['titles'] = title_query[0].get('id')
        else:
            db.query("INSERT INTO title_table (title) VALUES ('{}')".format(match_dict["titles"]))
            title_query = db.query("SELECT id FROM title_table WHERE title = '{}'".format(match_dict["titles"])).as_dict()
            match_insert_dict['titles'] = title_query[0].get('id')

        matchtype_query = db.query("SELECT id FROM matchtype_table WHERE matchtype = '{}'".format(match_dict["matchtype"])).as_dict()
        if matchtype_query:
            match_insert_dict['matchtype'] = matchtype_query[0].get('id')
        else:
            db.query("INSERT INTO matchtype_table (matchtype) VALUES ('{}')".format(match_dict["matchtype"]))
            matchtype_query = db.query("SELECT id FROM matchtype_table WHERE matchtype = '{}'".format(match_dict["matchtype"])).as_dict()
            match_insert_dict['matchtype'] = matchtype_query[0].get('id')

        db.query(match_insert_template.format(**match_insert_dict))

        team_insert_dict = {
            'match_id': match_dict['match_id'],
            'competitors': ''
        }

        winning_wrestler_id_list_for_team_id = []
        for winning_wrestler_url in match_dict['winners']:
            winning_wrestler_id = int(winning_wrestler_url.split('-')[-1])
            winning_wrestler_id_list_for_team_id.append(winning_wrestler_id)
        winning_wrestler_id_list_for_team_id.sort()
        winning_team_string = ''
        for winning_wrestler_id in winning_wrestler_id_list_for_team_id:
            winning_team_string += ''.join([wrestler_start_marker, str(winning_wrestler_id), wrestler_end_marker])
        competitor_string = ''.join([team_start_marker, winning_team_string, team_end_marker])
        for team_list in match_dict['teams']:
            team_id_list = []
            for wrestler_url in team_list:
                wrestler_id = int(wrestler_url.split('-')[-1])
                team_id_list.append(wrestler_id)
            team_id_list.sort()
            team_string = ''
            for id in team_id_list:
                if id in winning_wrestler_id_list_for_team_id:
                    break
                team_string += ''.join([wrestler_start_marker, str(id), wrestler_end_marker])
            if team_string:
                competitor_string += ''.join([team_start_marker, team_string, team_end_marker])

        team_insert_dict['competitors'] = competitor_string
        db.query(team_insert_template.format(**team_insert_dict))


def update_wrestler_table(wrestler_url_list, db, browser):
    for profightdb_id in wrestler_url_list:
        wrestler_query = db.query("SELECT id FROM wrestler_table WHERE profightdb_id = '{}'".format(profightdb_id)).as_dict()
        if not wrestler_query:
            # retry
            wrestler_query = db.query("SELECT id FROM wrestler_table WHERE profightdb_id = '{}'".format(profightdb_id)).as_dict()
            if not wrestler_query:

                insert_template = "INSERT INTO wrestler_table (profightdb_id, id, current_alias, nationality, dob) VALUES ('{profightdb_id}', {id}, '{current_alias}', '{nationality}', {dob})"
                insert_dict = {
                    'profightdb_id': profightdb_id,
                    'id': int(profightdb_id.split('-')[-1]),
                    'current_alias': '',
                    'nationality': '',
                    'dob': -1
                }
                browser.get(url='http://www.profightdb.com/wrestlers/{}.html'.format(profightdb_id))
                insert_dict['current_alias'] = clean_text(browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(1) > td:nth-child(3)').text[16:])
                nationality = browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(3) > td:nth-child(1)').text[13:]

                insert_dict['nationality'] = update_nationality_table(nationality=nationality, db=db)
                try:
                    date = browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(1) > a').text
                    month, day, year = date.split()
                    date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
                    insert_dict['dob'] = int(dt.datetime.strftime(date, "%Y%m%d"))
                except selenium.common.exceptions.NoSuchElementException:
                    insert_dict['dob'] = 0
                browser.back()

                db.query(insert_template.format(**insert_dict))


def match_parse(match_counter, browser, type, db):
    match_text_template =   "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({}) > a:nth-child({})"
    italic_text_template =  "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({}) > i > a:nth-child({})"

    if type == "winner":
        type_number = 2
    elif type == "loser":
        type_number = 4

    alias_url_dict = {}
    for counter in range(1, 75):
        try:
            link = browser.find_element_by_css_selector(match_text_template.format(
                match_counter, type_number, counter))
            url = link.get_attribute("href")[36:-5]
            alias = link.get_attribute("text")
            clean_alias = clean_text(alias)
            alias_url_dict[clean_alias] = url
        except selenium.common.exceptions.NoSuchElementException:
            try:
                link = browser.find_element_by_css_selector(italic_text_template.format(
                    match_counter, type_number, counter))
                url = link.get_attribute("href")[36:-5]
                alias = link.get_attribute("text")
                clean_alias = clean_text(alias)
                alias_url_dict[clean_alias] = url
            except selenium.common.exceptions.NoSuchElementException:
                break

    update_alias_table(alias_url_dict, db=db)

    just_url_list = list(alias_url_dict.values())
    return just_url_list


def update_alias_table(alias_and_url_dict, db):
    for alias, url in alias_and_url_dict.items():
        id = int(url.split('-')[-1])
        dupecheck = db.query("SELECT alias FROM alias_table WHERE alias = '{alias}'".format(alias=alias)).as_dict()
        if dupecheck:
            continue
        else:
            db.query("INSERT INTO alias_table (alias, id) VALUES ('{alias}', {id})".format(alias=alias, id=id))


def update_nationality_table(nationality, db):
    dupecheck = db.query("SELECT id FROM nationality_table WHERE nationality = '{nationality}'".format(nationality=nationality)).as_dict()
    if dupecheck:
        nationality_id = dupecheck[0]['id']
    else:
        db.query("INSERT INTO nationality_table (nationality) VALUES ('{nationality}')".format(nationality=nationality))
        nationality_query = db.query("SELECT id FROM nationality_table WHERE nationality = '{nationality}'".format(nationality=nationality)).as_dict()
        nationality_id = nationality_query[0]['id']

    return nationality_id


def clean_text(text):
    import unicodedata

    temp_string = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode()
    temp_string = temp_string.lower()
    temp_string = temp_string.replace("'", "")
    temp_string = temp_string.replace('"', "")
    temp_string = temp_string.replace('.', "")
    return temp_string


def last_date(card, db):
    lastdate_result = db.query("SELECT max(date) FROM match_table WHERE card = '{card}';".format(card=card), fetchall=True).as_dict()
    lastdate = lastdate_result[0].get('max(date)', None)
    if lastdate is None:
        lastdate = 20140101
    return lastdate


def last_page(card, browser):
    index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='
    last_page_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > div"

    browser.get(index_url_template.format(1, card))
    lastpage = browser.find_element_by_css_selector(last_page_template)
    lastpage = lastpage.get_attribute('innerHTML')
    lastpage = int(int(lastpage.split("showing 1-10 of ")[1][:-1])/10)+1
    lastpage = min(lastpage, 200)
    return lastpage


def vacuum(db=records.Database(db_url='sqlite:///G:/wrestlebot/wrestlebot.db'), verbose=False):
    if verbose:
        from os.path import getsize
        before = getsize(db.db_url[10:])
        db.query('VACUUM;')
        after = getsize(db.db_url[10:])
        percentage = 1 - after/before
        print('db cleanup resulted in a {percentage:.3%} reduction in db size'.format(percentage=percentage))

    else:
        db.query('VACUUM;')


if __name__ == '__main__':
    db = records.Database(db_url='sqlite:///G:/wrestlebot/wrestlebot.db')
    init = False
    verbose = True

    if init:
        table_dict_list = [
            {
                'table_name': 'match_table',
                'table_params': [
                    {
                        'key_name': 'match_id',
                        'key_type': 'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    },

                    {
                        'key_name': 'date',
                        'key_type': 'INTEGER',
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'duration',
                        'key_type':  'INTEGER',
                    },
                    {
                        'key_name': 'winner',
                        'key_type':  'INTEGER',
                    },
                    {
                        'key_name': 'wintype',
                        'key_type':  'INTEGER',
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'titles',
                        'key_type':  'INTEGER',
                    },
                    {
                        'key_name': 'matchtype',
                        'key_type':  'INTEGER',
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'card',
                        'key_type':  'TEXT',
                    },
                    {
                        'key_name': 'competitors',
                        'key_type': 'TEXT',
                    }
                ]
             },
            {
                'table_name': 'title_table',
                'table_params': [
                    {
                        'key_name': 'title',
                        'key_type':  'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    }
                ]
             },
            {
                'table_name': 'wintype_table',
                'table_params': [
                    {
                        'key_name': 'wintype',
                        'key_type': 'TEXT',
                        'unique_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    }
                ]
            },
            {
                'table_name': 'matchtype_table',
                'table_params': [
                    {
                        'key_name': 'matchtype',
                        'key_type':  'TEXT',
                        'unique_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    }
                ]
             },
            {
                'table_name': 'wrestler_table',
                'table_params': [
                    {
                        'key_name': 'profightdb_id',
                        'key_type':  'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    },

                    {
                        'key_name': 'current_alias',
                        'key_type':  'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'nationality',
                        'key_type': 'TEXT',
                    },
                    {
                        'key_name': 'dob',
                        'key_type': 'TEXT',
                    }
                ]
            },
            {
                'table_name': 'alias_table',
                'table_params': [
                    {
                        'key_name': 'alias',
                        'key_type':  'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                    },
                ]
            },
            {
                'table_name': 'nationality_table',
                'table_params': [
                    {
                        'key_name': 'nationality',
                        'key_type': 'TEXT',
                        'unique_flag': True,
                        'not_null_flag': True
                    },
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    },
                ]
            }

        ]

        for table_dict in table_dict_list:
            init_table(table_dict=table_dict, db=db)

        virtual_table_dict_list = [
            {
                'table_name': 'team_table',
                'table_params': [
                    {
                        'key_name': 'competitors',
                        'key_type': 'TEXT'
                    },
                    {
                        'key_name': 'match_id',
                        'key_type': 'TEXT'
                    }
                ]
             }
        ]
        for virtual_table_dict in virtual_table_dict_list:
            # init_virtual_table(table_dict=virtual_table_dict, db=db)    # disabled until i decide to implement FTS5; using LIKE instead until then.
            init_table(table_dict=virtual_table_dict, db=db)
    vacuum(db=db, verbose=verbose)

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)

    for card_dict in [{'name': 'wwe', 'number': 2}, {'name': 'nxt', 'number': 103}]:
        lastdate = last_date(card_dict['name'], db=db)
        lastpage = last_page(card_dict['number'], browser=browser)
        print("beginning {} scrape. last page is {}, last date was {}".format(card_dict['name'], lastpage, lastdate))
        full_parse(card_dict, lastdate=lastdate, lastpage=lastpage, browser=browser, force_update=False)
        vacuum(db=db, verbose=verbose)

    browser.close()
    print("finished!")
