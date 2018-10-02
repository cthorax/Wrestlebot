from app import db

class Nationality(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    wrestlers = db.relationship('Wrestler', backref='nationality', lazy='dynamic')

    def __repr__(self):
        return "<nationality '{}'>".format(self.name)

class Wintype(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    counts_as_win = db.Column(db.Boolean)
    matches = db.relationship('Matches', backref='wintype', lazy='dynamic')

    def __repr__(self):
        return "<wintype '{}'>".format(self.name)

class Matchtype(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    matches = db.relationship('Matches', backref='wintype', lazy='dynamic')

    def __repr__(self):
        return "<matchtype '{}'>".format(self.name)

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    arenas = db.relationship('Arena', backref='location', lazy='dynamic')

    def __repr__(self):
        return "<location '{}'>".format(self.name)

class Arena(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    location_id = db.Column(db.Integer, db.ForeignKey('location.id'))
    matches = db.relationship('Match', backref='arena', lazy='dynamic')

    def __repr__(self):
        return "<arena '{}'>".format(self.name)

class Wrestler(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    nationality_id = db.Column(db.Integer, db.ForeignKey('nationality.id'))
    matches = db.relationship('Match', backref='match', lazy='dynamic')

    def __repr__(self):
        return "<arena '{}'>".format(self.name)

class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=True)
    wrestlers = db.relationship('Wrestler', backref='member of', lazy='dynamic')
    # multiple wrestlers go here
    wrestler_id = db.Column(db.Integer, db.ForeignKey('wrestler.id'))

    def __repr__(self):
        if self.name:
            result = "<team '{}'>".format(self.name)
        else:
            team_name = ''
            # for wrestlers in team etc etc
            result = "<team '{}'>".format(team_name)
        return result

class Competition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teams = db.relationship('Team', backref='competed in', lazy='dynamic')
    matches = db.relationship('Matches', backref='wintype', lazy='dynamic')
    # multiple teams go here
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'))

    def __repr__(self):
        competition_name = ''
        # for teams in competition etc etc
        result = "<competition between '{}'>".format(competition_name)
        return result

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime)
    arena_id = db.Column(db.Integer, db.ForeignKey('arena.id'))
    competition_id = db.Column(db.Integer, db.ForeignKey('competition.id'))
    wintype_id = db.Column(db.Integer, db.ForeignKey('wintype.id'))
    matchtype_id = db.Column(db.Integer, db.ForeignKey('matchtype.id'))
