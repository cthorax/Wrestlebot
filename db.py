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
        ],
        'foreign_keys': [
            {
                'key_name': string required,
                'other_table': string required,
                'other_table_key': string required
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
    foreign_key_template = 'FOREIGN KEY({key_name}) REFERENCES {other_table}({other_table_key}),\n'

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

    foreign_key_queries = ''
    for foreign_key in table_dict.get('foreign_keys', []):
        foreign_key_query = foreign_key_template.format(**foreign_key)
        foreign_key_queries += foreign_key_query

    if foreign_key_queries:
        foreign_key_queries = foreign_key_queries[:-2]    # strips trailing ',\n'
        param_queries += foreign_key_queries
    else:
        param_queries = param_queries[:-2]  # strips trailing ',\n'

    create_query = create_template.format(table_name=table_dict.get('table_name'), table_params=param_queries)
    db.query(create_query)


def full_parse(card_dict, lastdate, lastpage, force_update=False):
    index_url_template = 'http://www.profightdb.com/cards/wwe-cards-pg{}-no-{}.html?order=&type='  # goes from 1 to whatever
    event_link_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(3) > a"
    index_date_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(1) > a"

    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
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

                    match_dict_list = page_parse(browser=browser, date=date, card_dict=card_dict)
                    update_match_table(match_dict_list=match_dict_list, db=db)
        else:
            print("all records up to date.")

        browser.close()

    except selenium.common.exceptions.WebDriverException:
        print("selenium window crashed")


def page_parse(browser, date, card_dict):
    match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({})"
    duration_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child(5)"

    url = browser.current_url[36:-5]
    match_dict_list = []
    for match_counter in range(2, 500):
        try:
            match_number = match_counter - 1
            wrestler_urls = []

            winner_urls = match_parse(match_counter, browser, type='winner')
            wrestler_urls.extend(winner_urls)

            wintype = browser.find_element_by_css_selector(
                match_text_template.format(match_counter, 3)).text
            wintype = wintype.lower()
            if not wintype:
                wintype = 'def.'  # only one blank wintype found so far but it was a huge PITA, so.

            loser_urls = match_parse(match_counter, browser, type='loser')
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

        except selenium.common.exceptions.NoSuchElementException:
            print("event \'{}\' done.".format(url))
            break

    browser.back()
    return match_dict_list


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
    insert_template = "INSERT INTO match_table (match_id, date, duration, winner, wintype, titles, matchtype, card, competitors) VALUES ('{match_id}', {date}, {duration}, {winner}, {wintype}, {titles}, {matchtype}, '{card}', {competitors})"

    for match_dict in match_dict_list:
        match_insert_dict = {
            'match_id': match_dict['match_id'],
            'date': match_dict['date'],
            'duration': match_dict['duration'],
            'winner': -1,
            'wintype': -1,
            'titles': -1,
            'matchtype': -1,
            'card': match_dict['card'],
            'competitors': -1
        }

        for wrestler_url in match_dict['wrestlers']:
            wrestler_query = db.query("SELECT id FROM wrestler_table WHERE profightdb_id = '{}'".format(wrestler_url)).as_dict()
            if not wrestler_query:
                # retry
                id = int(wrestler_url.split('-')[-1])
                wrestler_query = db.query("SELECT id FROM wrestler_table WHERE id = {}".format(id)).as_dict()
                if not wrestler_query:
                    update_wrestler_table(profightdb_id=wrestler_url, db=db)
                
        team_id_list_for_competitor_id = []
        for team_list in match_dict['teams']:
            wrestler_id_list_for_team_id = []
            for wrestler_url in team_list:
                wrestler_query = db.query("SELECT id FROM wrestler_table WHERE profightdb_id = '{}'".format(wrestler_url)).as_dict()
                wrestler_id = wrestler_query[0].get('id')
                wrestler_id_list_for_team_id.append(wrestler_id)
            wrestler_id_list_for_team_id.sort()     # prevent combinations by sorting
            team_members_string = ''
            for number, wrestler_id in enumerate(wrestler_id_list_for_team_id):
                if team_members_string:
                    team_members_string += " AND wrestler{number} = {wrestler_id}".format(number=str(number).zfill(2), wrestler_id=wrestler_id)
                else:
                    team_members_string = "wrestler00 = {}".format(wrestler_id)
            team_members_string += ' AND wrestler{} is NULL;'.format(str(len(wrestler_id_list_for_team_id)).zfill(2))

            if len(wrestler_id_list_for_team_id)+1 > max_team_size:
                update_team_size(new_value=len(wrestler_id_list_for_team_id)+1, db=db)

            team_query = db.query('SELECT id FROM team_table WHERE {}'.format(team_members_string)).as_dict()
            if team_query:
                team_id = team_query[0].get('id')
            else:
                update_team_table(team_id_list=wrestler_id_list_for_team_id, db=db)
                team_query = db.query('SELECT id FROM team_table WHERE {}'.format(team_members_string)).as_dict()
                team_id = team_query[0].get('id')

            team_id_list_for_competitor_id.append(team_id)
        team_id_list_for_competitor_id.sort()     # prevent combinations by sorting

        competitor_composition_string = ''
        for number, team_id in enumerate(team_id_list_for_competitor_id):
            if competitor_composition_string:
                competitor_composition_string += ' AND team{number} = {team_id}'.format(
                    number=str(number).zfill(2), team_id=team_id)
            else:
                competitor_composition_string = 'team00 = {}'.format(team_id)
        competitor_composition_string += ' AND team{} is NULL;'.format(str(len(team_id_list_for_competitor_id)).zfill(2))

        if len(team_id_list_for_competitor_id)+1 > max_competitors:
            update_competitor_size(new_value=len(team_id_list_for_competitor_id)+1, db=db)

        competitor_query = db.query('SELECT id FROM competitors_table WHERE {}'.format(competitor_composition_string)).as_dict()
        if competitor_query:
            competitor_id = competitor_query[0].get('id')
        else:
            update_competitors_table(competitor_id_list=team_id_list_for_competitor_id, db=db)
            competitor_query = db.query('SELECT id FROM competitors_table WHERE {}'.format(competitor_composition_string)).as_dict()
            competitor_id = competitor_query[0].get('id')

        match_insert_dict['competitors'] = competitor_id
        
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

        winning_wrestler_id_list_for_team_id = []
        for winning_wrestler_url in match_dict['winners']:
            winning_wrestler_query = db.query("SELECT id FROM wrestler_table WHERE profightdb_id = '{}'".format(winning_wrestler_url))
            winning_wrestler_id_list_for_team_id.append(winning_wrestler_query[0].get('id'))
        winning_wrestler_id_list_for_team_id.sort()  # prevent combinations by sorting
        winning_team_members_string = ''
        for winning_number, winning_wrestler_id in enumerate(winning_wrestler_id_list_for_team_id):
            if winning_team_members_string:
                winning_team_members_string += " AND wrestler{number} = {wrestler_id}".format(
                    number=str(winning_number).zfill(2), wrestler_id=winning_wrestler_id)
            else:
                winning_team_members_string = 'wrestler00 = {}'.format(winning_wrestler_id)
        winning_team_members_string += ' AND wrestler{} IS NULL;'.format(str(len(winning_wrestler_id_list_for_team_id)).zfill(2))

        if len(winning_wrestler_id_list_for_team_id)+1 > max_team_size:
            update_team_size(new_value=len(winning_wrestler_id_list_for_team_id)+1, db=db)

        winning_team_query = db.query('SELECT id FROM team_table WHERE {}'.format(winning_team_members_string)).as_dict()
        try:
            winning_team_id = winning_team_query[0].get('id')
        except IndexError:
            pass
        match_insert_dict['winner'] = winning_team_id

        db.query(insert_template.format(**match_insert_dict))


def update_competitors_table(competitor_id_list, db):
    insert_template = "INSERT INTO competitors_table ({columns}) VALUES ({values})"
    columns = ''
    values = ''
    for number, team_id in enumerate(competitor_id_list):
        if columns:
            columns += ', team{number}'.format(number=str(number).zfill(2))
        if values:
            values += ', {team_id}'.format(team_id=team_id)
        else:
            values = '{team_id}'.format(team_id=team_id)
            columns = 'team00'
    db.query(insert_template.format(columns=columns, values=values))


def update_team_table(team_id_list, db):
    insert_template = "INSERT INTO team_table ({columns}) VALUES ({values})"
    columns = ''
    values = ''
    for number, wrestler_id in enumerate(team_id_list):
        if columns:
            columns += ', wrestler{number}'.format(number=str(number).zfill(2))
        if values:
            values += ', {wrestler_id}'.format(wrestler_id=wrestler_id)
        else:
            values = '{wrestler_id}'.format(wrestler_id=wrestler_id)
            columns = 'wrestler00'
    db.query(insert_template.format(columns=columns, values=values))


def update_wrestler_table(profightdb_id, db):
    insert_template = "INSERT INTO wrestler_table (profightdb_id, id, current_alias, nationality, dob) VALUES ('{profightdb_id}', {id}, '{current_alias}', '{nationality}', {dob})"
    insert_dict = {
        'profightdb_id': profightdb_id,
        'id': int(profightdb_id.split('-')[-1]),
        'current_alias': '',
        'nationality': '',
        'dob': -1
    }
    browser = webdriver.Chrome("chromedriver.exe", chrome_options=chrome_options)
    browser.get(url='http://www.profightdb.com/wrestlers/{}.html'.format(profightdb_id))
    insert_dict['current_alias'] = clean_text(browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(1) > td:nth-child(3)').text[16:])
    insert_dict['nationality'] = browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(3) > td:nth-child(1)').text[13:]
    try:
        date = browser.find_element_by_css_selector('body > div > div.wrapper > div.content-wrapper > div.content > div:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(1) > a').text
        month, day, year = date.split()
        date = dt.datetime.strptime("{} {} {}".format(day[:-2], month, year), "%d %b %Y")
        insert_dict['dob'] = int(dt.datetime.strftime(date, "%Y%m%d"))
    except selenium.common.exceptions.NoSuchElementException:
        insert_dict['dob'] = 0
    browser.close()

    db.query(insert_template.format(**insert_dict))


def match_parse(match_counter, browser, type):
    other_match_text_template = "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({}) > a:nth-child({})"
    alternate_text_template =   "body > div > div.wrapper > div.content-wrapper > div.content.inner > div.right-content > div > div > table > tbody > tr:nth-child({}) > td:nth-child({}) > i > a:nth-child({})"

    if type == "winner":
        type_number = 2
    elif type == "loser":
        type_number = 4

    temp_urls = []
    for counter in range(1, 75):
        try:
            url = browser.find_element_by_css_selector(other_match_text_template.format(
                match_counter, type_number, counter)).get_attribute("href")[36:-5]
            temp_urls.append(url)
        except selenium.common.exceptions.NoSuchElementException:
            try:
                url = browser.find_element_by_css_selector(alternate_text_template.format(
                    match_counter, type_number, counter)).get_attribute("href")[36:-5]
                temp_urls.append(url)
            except selenium.common.exceptions.NoSuchElementException:
                break

    return temp_urls


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


def update_team_size(new_value, db=records.Database(db_url='sqlite:///G:/wrestlebot/wrestlebot.db'), verbose=False):
    import sqlalchemy
    global max_team_size
    for number in range(max_team_size, new_value):
        formatted_number = str(number).zfill(2)
        try:
            db.query('ALTER TABLE team_table ADD COLUMN wrestler{} INTEGER'.format(formatted_number))
        except sqlalchemy.exc.OperationalError:
            pass

    max_team_size = new_value


def update_competitor_size(new_value, db=records.Database(db_url='sqlite:///G:/wrestlebot/wrestlebot.db'), verbose=False):
    import sqlalchemy
    global max_competitors
    for number in range(max_competitors, new_value):
        formatted_number = str(number).zfill(2)
        try:
            db.query('ALTER TABLE competitors_table ADD COLUMN team{} INTEGER'.format(formatted_number))
        except sqlalchemy.exc.OperationalError:
            pass

    max_competitors = new_value


if __name__ == '__main__':
    db = records.Database(db_url='sqlite:///G:/wrestlebot/wrestlebot.db')
    init = False
    verbose = True

    global max_team_size
    global max_competitors
    max_team_size = 1
    max_competitors = 1

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
                        'key_type': 'INTEGER',
                    }
                ],
                'foreign_keys': [
                    {
                        'key_name': 'winner',
                        'other_table': 'team_table',
                        'other_table_key': 'id'
                    },
                    {
                        'key_name': 'titles',
                        'other_table': 'title_table',
                        'other_table_key': 'id'
                    },
                    {
                        'key_name': 'matchtype',
                        'other_table': 'matchtype_table',
                        'other_table_key': 'id'
                    },
                    {
                        'key_name': 'competitors',
                        'other_table': 'competitors_table',
                        'other_table_key': 'id'
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
            }
        ]
        team_table_dict = {
                'table_name': 'team_table',
                'table_params': [
                    {
                        'key_name': 'id',
                        'key_type': 'INTEGER',
                        'primary_key_flag': True
                    }
                ]
            }
        for number in range(max_team_size):
            params_dict = {
                    'key_name': 'wrestler{}'.format(str(number).zfill(2)),
                    'key_type': 'INTEGER'
                }
            team_table_dict['table_params'].append(params_dict)
        table_dict_list.append(team_table_dict)

        competitors_table_dict = {
            'table_name': 'competitors_table',
            'table_params': [
                {
                    'key_name': 'id',
                    'key_type': 'INTEGER',      # pretty quickly gets larger than 2^64, which is the INTEGER limit.
                    'primary_key_flag': True
                }
            ],
            'foreign_keys': []
        }
        for number in range(max_competitors):
            params_dict = {
                    'key_name': 'team{}'.format(str(number).zfill(2)),
                    'key_type': 'INTEGER'
                }
            competitors_table_dict['table_params'].append(params_dict)
        table_dict_list.append(competitors_table_dict)

        for table_dict in table_dict_list:
            init_table(table_dict=table_dict, db=db)
    vacuum(db=db, verbose=verbose)

    matches = []
    for card_dict in [{'name': 'wwe', 'number': 2}, {'name': 'nxt', 'number': 103}]:
        lastdate = last_date(card_dict['name'], db=db)
        lastpage = last_page(card_dict['number'])
        print("beginning {} scrape. last page is {}, last date was {}".format(card_dict['name'], lastpage, lastdate))
        full_parse(card_dict, lastdate=lastdate, lastpage=lastpage, force_update=False)
        vacuum(db=db, verbose=verbose)