def mapping():

    map_values_serie_a = {
        'Palmeiras': 'Palmeiras',
        'br Palmeiras': 'Palmeiras',
        'Internacional': 'Internacional',
        'Atlético Mineiro': 'Atletico Mineiro',
        'br Atlético Mineiro': 'Atletico Mineiro',
        'São Paulo': 'Sao Paulo',
        'Santos': 'Santos',
        'Grêmio': 'Gremio',
        'Goiás': 'Goias',
        'Fortaleza': 'Fortaleza',
        'América (MG)': 'America MG',
        'Cuiabá': 'Cuiaba',
        'Cruzeiro': 'Cruzeiro',
        'Coritiba': 'Coritiba',
        'Corinthians': 'Corinthians',
        'Bragantino': 'Bragantino',
        'Botafogo (RJ)': 'Botafogo RJ',
        'Bahia': 'Bahia'
    }

    map_values_serie_b = {
        'Sampaio Corrêa' : 'Sampaio Correa',
        'Náutico' : 'Nautico',
        'Operário' : 'Operario',
        'Avaí' : 'Avai',
        'Confiança' : 'Confianca',
        'Grêmio Novorizontino' : 'Gremio Novorizontino',
        'Botafogo (SP)' : 'Botafogo SP',
        'Criciúma' : 'Criciuma',
        'Grêmio' : 'Gremio',
        'Cuiabá' : 'Cuiaba',
        'América (MG)' : 'America MG',
        'Goiás' : 'Goias',
        'Botafogo (RJ)' : 'Botafogo RJ',
        'Atl Goianiense' : 'Atletico Goianiense'
} 

    map_values_argentina = {
        'Vélez Sarsf': 'Velez Sarsf',
        'CA Huracán': 'CA Huracan',
        'Colón': 'Colon',
        'Lanús': 'Lanus',
        "Newell's OB": 'Newells OB',
        'CA Unión': 'CA Union',
        'Arsenal': 'Arsenal',
        'Tucumán': 'Tucuman',
        'San Martín SJ': 'San Martin SJ',
        'CC Córdoba': 'CC Cordoba'

    }

    map_values_jpn = {
        'Urawa Reds': 'Urawa Red Diamonds',
        'Sanfrecce': 'Sanfrecce Hiroshima',
        'Marinos': 'Yokohama F Marinos',
        'Kawa Frontale': 'Kawasaki Frontale',
        'Grampus': 'Nagoya Grampus',
        'S-Pulse': 'Shimizu S Pulse',
        'Consa Sapporo': 'Hokkaido Consadole Sapporo',
        'Júbilo Iwata': 'Jubilo Iwata',
        'Yamaga': 'Matsumoto Yamaga',
        'Vortis': 'Tokushima Vortis',
        'V-V Nagasaki': 'V Varen Nagasaki',
        'Montedio': 'Montedio Yamagata'
    }

    map_values_norway = {
        'Vålerenga': 'Valerenga',
        'Strømsgodset': 'Stromsgodset',
        'Lillestrøm': 'Lillestrom',
        'Stabæk': 'Stabaek',
        'Bodø/Glimt': 'BodoGlimt',
        'Tromsø': 'Tromso',
        'Mjøndalen': 'Mjondalen'
    }

    map_values_finland = {
        'HJK': 'HJK Helsinki',
        'HIFK': 'Helsinki IFK',
        'Kemi Kings': 'PS Kemi Kings',
        'JJK': 'JJK Jyvaskyla',
        'PK–35': 'PK 35 Vantaa'
    }
    
    # map_values_england = {
    #     "Brighton and Hove Albion": "Brighton", 
    #     "Manchester United": "Manchester Utd", 
    #     "Newcastle United": "Newcastle Utd", 
    #     "Tottenham Hotspur": "Tottenham", 
    #     "West Ham United": "West Ham", 
    #     "Wolverhampton Wanderers": "Wolves"
    # } 

    map_values_england = {
        'Brighton': 'Brighton and Hove Albion',
        'Manchester Utd': 'Manchester United',
        'Newcastle Utd': 'Newcastle United',
        'Tottenham': 'Tottenham Hotspur',
        'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton Wanderers'
    }


    map_values_italy = {
        'Internazionale' : 'Inter'
    }

    map_values_spain = {
        "Atlético Madrid": "Atletico Madrid",
        "Athletic Club": "Athletic Club",
        "Betis": "Real Betis",
        "Málaga": "Malaga",
        "Alavés": "Alaves",
        "Eibar": "Eibar",
        "La Coruña": "Deportivo La Coruna",
        "Leganés": "Leganes",
        "Las Palmas": "Las Palmas",
        "Cádiz": "Cadiz",
        "Almería": "Almeria",
        "Sporting Gijón": "Sporting Gijon",
        "Córdoba": "Cordoba",
        "Racing Sant": "Racing Santander"
    }

    
    map_values_germany = {
        "Leverkusen": "Bayer Leverkusen",
        "M'Gladbach": "Monchengladbach",
        "Eint Frankfurt": "Eintracht Frankfurt",
        "Köln": "Koln",
        "Nürnberg": "Nurnberg",
        "Greuther Fürth": "Greuther Furth",
        "Braunschweig": "Eintracht Braunschweig"
        }

    map_values_france = {
        "Paris S-G": "Paris Saint Germain",
        "Saint-Étienne": "Saint Etienne",
        "Gazélec Ajaccio": "Gazelec Ajaccio"
        }

    map_values_greece = {
        "Iraklis" : "Iraklis Thessaloniki" 
        }
    
    mapped_values = [map_values_serie_a, map_values_serie_b, map_values_argentina, 
                    map_values_jpn, map_values_norway, map_values_finland,
                    map_values_england, map_values_italy, map_values_spain,
                    map_values_germany, map_values_france, map_values_greece]
    
    return mapped_values