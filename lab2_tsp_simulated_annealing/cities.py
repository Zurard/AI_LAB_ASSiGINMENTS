# 22 important tourist locations in Rajasthan.
# Coordinates are approximate city-centre latitudes and longitudes (WGS84).
# Cost of travel is taken as the great-circle distance between two points.

CITIES = [
    # name,                  latitude,  longitude
    ("Jaipur",               26.9124,   75.7873),   # Pink City, Amber Fort, Hawa Mahal
    ("Ajmer",                26.4499,   74.6399),   # Dargah Sharif
    ("Pushkar",              26.4897,   74.5510),   # Brahma Temple, Pushkar Lake
    ("Jodhpur",              26.2389,   73.0243),   # Blue City, Mehrangarh Fort
    ("Jaisalmer",            26.9157,   70.9083),   # Golden City, desert fort
    ("Bikaner",              28.0229,   73.3119),   # Junagarh Fort
    ("Udaipur",              24.5854,   73.7125),   # City of Lakes
    ("Mount Abu",            24.5926,   72.7156),   # Dilwara Temples, Nakki Lake
    ("Chittorgarh",          24.8887,   74.6269),   # Chittorgarh Fort
    ("Kumbhalgarh",          25.1528,   73.5870),   # Kumbhalgarh Fort
    ("Ranakpur",             25.1156,   73.4722),   # Jain marble temples
    ("Nathdwara",            24.9381,   73.8236),   # Shrinathji Temple
    ("Ranthambore",          26.0173,   76.3559),   # tiger reserve (Sawai Madhopur)
    ("Bundi",                25.4305,   75.6499),   # Taragarh Fort, stepwells
    ("Kota",                 25.2138,   75.8648),   # Chambal palaces
    ("Alwar",                27.5530,   76.6346),   # Bala Quila, City Palace
    ("Bharatpur",            27.2173,   77.4895),   # Keoladeo National Park
    ("Mandawa",              28.0556,   75.1481),   # Shekhawati havelis
    ("Nagaur",               27.2020,   73.7339),   # Nagaur Fort
    ("Osian",                26.7272,   72.8992),   # desert temples
    ("Barmer",               25.7521,   71.3966),   # desert crafts
    ("Dungarpur",            23.8430,   73.7147),   # Juna Mahal
]


def names():
    return [c[0] for c in CITIES]


def coords():
    return [(c[1], c[2]) for c in CITIES]
