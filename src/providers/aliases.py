"""Mapping provider team names to Football-Data team names.

Football-Data names are the canonical names of the system. A provider name
is resolved in two controlled steps, never by fuzzy matching:

1. explicit alias table ``TEAM_ALIASES`` (normalised provider name ->
   Football-Data name);
2. exact match of the normalised names inside the same league
   (e.g. "Nottingham Forest" vs "Nott'm Forest" needs an alias,
   "Bournemouth" vs "AFC Bournemouth" does not).

Unresolved names are returned as None; the caller skips the fixture and
logs the name so an alias can be added here.
"""
from __future__ import annotations

import re
import unicodedata

# Words that never distinguish two clubs of one league.
_NOISE = {"fc", "afc", "cf", "sc", "ac", "cd", "ud", "sd", "rc", "rcd", "ca", "club", "de", "the",
          "calcio", "sv", "vfb", "vfl", "tsg", "fk", "sk", "as", "ssc", "us", "sl", "kv", "krc",
          "kaa", "rsc", "royal", "rfc", "jk", "bk", "if", "aj", "ogc", "stade", "olympique",
          "football", "fussball", "1"}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("&", " and "))
    words = [w for w in text.split() if w not in _NOISE]
    return " ".join(words)


# normalised provider name -> Football-Data name
TEAM_ALIASES: dict[str, str] = {
    # England
    "manchester united": "Man United", "manchester utd": "Man United",
    "manchester city": "Man City",
    "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
    "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
    "tottenham hotspur": "Tottenham", "tottenham": "Tottenham",
    "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "west ham united": "West Ham", "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton",
    "leeds united": "Leeds", "leicester city": "Leicester", "ipswich town": "Ipswich",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds", "west bromwich albion": "West Brom", "west brom": "West Brom",
    "queens park rangers": "QPR", "blackburn rovers": "Blackburn", "bolton wanderers": "Bolton",
    "preston north end": "Preston", "stoke city": "Stoke", "hull city": "Hull", "swansea city": "Swansea",
    "cardiff city": "Cardiff", "coventry city": "Coventry", "norwich city": "Norwich", "luton town": "Luton",
    "derby county": "Derby", "oxford united": "Oxford", "plymouth argyle": "Plymouth",
    "bristol city": "Bristol City", "birmingham city": "Birmingham", "charlton athletic": "Charlton",
    "wycombe wanderers": "Wycombe", "peterborough united": "Peterboro", "peterborough": "Peterboro",
    "milton keynes dons": "Milton Keynes Dons", "mk dons": "Milton Keynes Dons",
    "huddersfield town": "Huddersfield", "rotherham united": "Rotherham", "lincoln city": "Lincoln",
    "cambridge united": "Cambridge", "exeter city": "Exeter", "stockport county": "Stockport",
    "wigan athletic": "Wigan", "burton albion": "Burton", "shrewsbury town": "Shrewsbury",
    "mansfield town": "Mansfield", "northampton town": "Northampton", "leyton orient": "Leyton Orient",
    "crawley town": "Crawley Town", "doncaster rovers": "Doncaster", "bradford city": "Bradford",
    "port vale": "Port Vale", "notts county": "Notts County", "grimsby town": "Grimsby",
    "salford city": "Salford", "harrogate town": "Harrogate", "accrington stanley": "Accrington",
    "tranmere rovers": "Tranmere", "newport county": "Newport County", "swindon town": "Swindon",
    "colchester united": "Colchester", "carlisle united": "Carlisle", "chesterfield": "Chesterfield",
    "barrow": "Barrow", "fleetwood town": "Fleetwood Town", "gillingham": "Gillingham",
    "cheltenham town": "Cheltenham", "crewe alexandra": "Crewe", "bromley": "Bromley",
    "afc wimbledon": "AFC Wimbledon", "wimbledon": "AFC Wimbledon",
    "bristol rovers": "Bristol Rvs", "barnsley": "Barnsley", "reading": "Reading",
    "stevenage": "Stevenage", "blackpool": "Blackpool", "wrexham": "Wrexham",
    # Scotland
    "glasgow rangers": "Rangers", "heart of midlothian": "Hearts", "hearts": "Hearts",
    "hibernian": "Hibernian", "st mirren": "St Mirren", "st johnstone": "St Johnstone",
    "dundee united": "Dundee United", "inverness ct": "Inverness C", "inverness caledonian thistle": "Inverness C",
    "queen of the south": "Queen of Sth", "airdrieonians": "Airdrie Utd", "raith rovers": "Raith Rvs",
    "greenock morton": "Morton", "partick thistle": "Partick", "dunfermline athletic": "Dunfermline",
    "ayr united": "Ayr", "hamilton academical": "Hamilton", "queens park": "Queens Park",
    # Spain
    "atletico madrid": "Ath Madrid", "atletico de madrid": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic": "Ath Bilbao",
    "real betis": "Betis", "betis": "Betis", "celta vigo": "Celta", "celta": "Celta",
    "espanyol": "Espanol", "rayo vallecano": "Vallecano", "real sociedad": "Sociedad",
    "deportivo alaves": "Alaves", "alaves": "Alaves", "real valladolid": "Valladolid",
    "real oviedo": "Oviedo", "sporting gijon": "Sp Gijon", "racing santander": "Santander",
    "deportivo la coruna": "La Coruna", "deportivo": "La Coruna", "real zaragoza": "Zaragoza",
    # Italy
    "inter": "Inter", "internazionale": "Inter", "ac milan": "Milan", "milan": "Milan",
    "as roma": "Roma", "hellas verona": "Verona", "verona": "Verona",
    # Germany
    "bayern munich": "Bayern Munich", "bayern munchen": "Bayern Munich", "bayern": "Bayern Munich",
    "borussia dortmund": "Dortmund", "dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach", "monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt", "bayer leverkusen": "Leverkusen", "leverkusen": "Leverkusen",
    "rb leipzig": "RB Leipzig", "leipzig": "RB Leipzig", "fsv mainz 05": "Mainz", "mainz 05": "Mainz",
    "mainz": "Mainz", "koln": "FC Koln", "cologne": "FC Koln", "hoffenheim": "Hoffenheim",
    "werder bremen": "Werder Bremen", "union berlin": "Union Berlin", "hamburger": "Hamburg",
    "hamburg": "Hamburg", "st pauli": "St Pauli", "heidenheim": "Heidenheim",
    "augsburg": "Augsburg", "wolfsburg": "Wolfsburg", "freiburg": "Freiburg",
    "stuttgart": "Stuttgart", "bochum": "Bochum", "hertha berlin": "Hertha", "hertha bsc": "Hertha",
    "fortuna dusseldorf": "Fortuna Dusseldorf", "greuther furth": "Greuther Furth",
    "schalke 04": "Schalke 04", "schalke": "Schalke 04", "nurnberg": "Nurnberg",
    "kaiserslautern": "Kaiserslautern", "karlsruher": "Karlsruhe", "darmstadt 98": "Darmstadt",
    "paderborn 07": "Paderborn", "hannover 96": "Hannover", "magdeburg": "Magdeburg",
    "elversberg": "Elversberg", "preussen munster": "Preußen Münster", "holstein kiel": "Holstein Kiel",
    "arminia bielefeld": "Bielefeld", "eintracht braunschweig": "Braunschweig",
    # France
    "paris saint germain": "Paris SG", "psg": "Paris SG", "marseille": "Marseille",
    "lyon": "Lyon", "saint etienne": "St Etienne", "st etienne": "St Etienne",
    "rennes": "Rennes", "stade rennais": "Rennes", "brest": "Brest", "stade brestois 29": "Brest",
    "monaco": "Monaco", "lille": "Lille", "lens": "Lens", "nice": "Nice", "nantes": "Nantes",
    "strasbourg": "Strasbourg", "reims": "Reims", "toulouse": "Toulouse", "montpellier": "Montpellier",
    "angers": "Angers", "auxerre": "Auxerre", "le havre": "Le Havre", "lorient": "Lorient",
    "metz": "Metz", "paris": "Paris FC",
    # Netherlands / Belgium / Portugal / Turkey / Greece
    "psv eindhoven": "PSV Eindhoven", "psv": "PSV Eindhoven", "az alkmaar": "AZ Alkmaar", "az": "AZ Alkmaar",
    "feyenoord": "Feyenoord", "ajax": "Ajax", "twente": "Twente", "utrecht": "Utrecht",
    "nec nijmegen": "Nijmegen", "nec": "Nijmegen", "go ahead eagles": "Go Ahead Eagles",
    "sparta rotterdam": "Sparta Rotterdam", "heerenveen": "Heerenveen", "fortuna sittard": "For Sittard",
    "pec zwolle": "Zwolle", "zwolle": "Zwolle", "groningen": "Groningen", "heracles": "Heracles",
    "nac breda": "NAC Breda", "willem ii": "Willem II", "excelsior": "Excelsior", "telstar": "Telstar",
    "volendam": "Volendam",
    "club brugge": "Club Brugge", "club brugge kv": "Club Brugge", "union st gilloise": "St. Gilloise",
    "union saint gilloise": "St. Gilloise", "anderlecht": "Anderlecht", "genk": "Genk", "gent": "Gent",
    "standard liege": "Standard", "standard": "Standard", "antwerp": "Antwerp", "cercle brugge": "Cercle Brugge",
    "westerlo": "Westerlo", "mechelen": "Mechelen", "charleroi": "Charleroi", "sint truiden": "St Truiden",
    "st truiden": "St Truiden", "oh leuven": "Oud-Heverlee Leuven", "oud heverlee leuven": "Oud-Heverlee Leuven",
    "dender": "Dender", "la louviere": "RAAL La Louviere", "zulte waregem": "Waregem",
    "sporting cp": "Sp Lisbon", "sporting lisbon": "Sp Lisbon", "sporting": "Sp Lisbon",
    "benfica": "Benfica", "porto": "Porto", "braga": "Sp Braga", "sporting braga": "Sp Braga",
    "vitoria guimaraes": "Guimaraes", "guimaraes": "Guimaraes", "famalicao": "Famalicao",
    "gil vicente": "Gil Vicente", "estoril": "Estoril", "casa pia": "Casa Pia", "moreirense": "Moreirense",
    "rio ave": "Rio Ave", "arouca": "Arouca", "santa clara": "Santa Clara", "nacional": "Nacional",
    "estrela amadora": "Estrela", "alverca": "Alverca", "tondela": "Tondela", "avs": "AVS",
    "galatasaray": "Galatasaray", "fenerbahce": "Fenerbahce", "besiktas": "Besiktas",
    "trabzonspor": "Trabzonspor", "istanbul basaksehir": "Buyuksehyr", "basaksehir": "Buyuksehyr",
    "olympiakos piraeus": "Olympiakos", "olympiacos": "Olympiakos", "panathinaikos": "Panathinaikos",
    "aek athens": "AEK", "paok": "PAOK", "aris": "Aris",
}


def resolve(name: str, known: set[str] | None = None) -> str | None:
    """Football-Data name for a provider team name, or None if unknown.

    ``known`` = Football-Data team names of the league (used for the exact
    normalised match and to reject aliases pointing outside the league).
    """
    key = normalise(name)
    alias = TEAM_ALIASES.get(key)
    if alias is not None and (known is None or alias in known):
        return alias
    if known:
        by_norm = {normalise(k): k for k in known}
        if key in by_norm:
            return by_norm[key]
    return None
