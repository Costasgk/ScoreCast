import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from Helper import mapping as mapped_values

class MissingDict(dict):
    __missing__ = lambda self, key: key

def read_train(path):
    df_training = pd.read_csv(path, index_col=0)
    return df_training

def read_test(path):
    df_testing = pd.read_csv(path, index_col= 0)
    return df_testing

def create_predictors(df_train, df_test=None):
    df_train['date'] = pd.to_datetime(df_train['date'])

    df_train['venue_code'] = df_train['venue'].astype('category').cat.codes
    df_train['opp_code'] = df_train['opponent'].astype('category').cat.codes

    df_train['hour'] = df_train['time'].str.replace(":.+", "",regex=True)
    df_train['hour'] = df_train['hour'].fillna(df_train['hour'].value_counts().idxmax())

    df_train['day_code'] = df_train['date'].dt.dayofweek 
    df_train['day_code'] = df_train['day_code'].fillna(df_train['day_code'].value_counts().idxmax())

    # df_train['target'] = (df_train['result'] == 'W').astype('int')
    result_mapping = {'L': 0, 'W': 1, 'D': 2}

    df_train['target'] = df_train['result'].map(result_mapping)

    if df_test != None:
        df_test['date'] = pd.to_datetime(df_test['date'])

        df_test['venue_code'] = df_test['venue'].astype('category').cat.codes

        df_test['opp_code'] = df_test['opponent'].astype('category').cat.codes

        df_test['hour'] = df_test['time'].str.replace(":.+", "",regex=True)
        df_test['hour'] = df_test['hour'].fillna(df_test['hour'].value_counts().idxmax())

        df_test['day_code'] = df_test['date'].dt.dayofweek 

        # df_test['target'] = (df_test['result'] == 'W').astype('int')
        df_test['target'] = df_test['result'].map(result_mapping)

        return df_train, df_test

    return df_train


# Calculate the rolling average for the last 5 games
# and replace NaN values with the rolling average.
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(5, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def update_with_rolling_average(df, cols, new_cols):
    matches_rolling = df.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    return matches_rolling

def combine(combined, matches_rolling):
    combined_data = combined.merge(matches_rolling[['date', 'time', 'team', 'opponent', 'result']],
                              left_index = True, right_index = True)
    return combined_data

def train_model(data, predictors, url):
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    train = data[data["date"] <= '2023-09-02']
    test = data[data["date"] > '2023-09-02']
    if url == url_arg:
        train = data[data["date"] <= '2023-07-19']
        test = data[data["date"] > '2023-07-23']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    # error = precision_score(test["target"], preds)
    return rf, combined

def map_values(combined_df):
    
    # mapping_values = [map_values_serie_a, map_values_serie_b, map_values_argentina, map_values_jpn, map_values_norway, map_values_finland]
    mapping_values = mapped_values()

    for map_values in mapping_values:

        # combined_df["new_team"] = combined_df["team"].map(map_values)

        mapping = MissingDict(**map_values)

        combined_df["new_team"] = combined_df["team"].map(mapping)
        
        merged = combined_df.merge(combined_df, left_on=["date", 'opponent'], right_on=["date", 'new_team'])

        merged = merged[['date', 'time_x', 'new_team_x', 'new_team_y', 'predicted_x', 'predicted_y']]

        map_string = {0: 'L', 1 : 'W', 2 : 'D'}
        
        merged['new_predicted_x'] = merged['predicted_x'].map(map_string)
        merged['new_predicted_y'] = merged['predicted_y'].map(map_string)
        
        renamed_columns = {
            'date' : 'Date',
            'time_x' : 'Time',
            'new_team_x' : 'Team A',
            'new_team_y' : 'Team B',
            'new_predicted_x' : 'Prediction for Team A',
            'new_predicted_y' : 'Prediction for Team B'
        }

        merged.rename(columns=renamed_columns, inplace=True)
        
        merged = merged.drop(['predicted_x', 'predicted_y'], axis = 1)
    
    return merged

def drop_teams(df, url):

    # Premiera Division Argentina
    teams_to_drop_arg = [
                'uy Nacional', 'pe Sporting Cristal', 'co Santa Fe', 'br São Paulo', 'br Santos', 
                'ec Independiente', 'br Flamengo', 'ec Barcelona SC', 'br Bragantino', 'co Nacional', 
                'br Corinthians', 'cl Palestino', 'py Guaraní', 'ar River Plate', 'ec LDU de Quito', 
                'br Fluminense', 'ar Vélez Sarsf', 'uy Wanderers', 'br Fortaleza', 'py Libertad', 
                'bo The Strongest', 'br Grêmio', 'br Chapecoense', 'cl La Calera', 'cl Huachipato', 
                'cl Univ Católica', 'br Ceará', 've Dep La Guaira', 'co Junior', 'br Palmeiras', 
                'ar San Lorenzo', 'py Olimpia', 'py 12 de Octubre', 'ar Estudiantes', 'br Atl Goianiense', 
                'br Atlético Mineiro', 'ar Talleres', 'cl Colo-Colo', 'ar Lanús', 'uy Danubio', 
                'cl Audax Italiano', 'br Cruzeiro', 'mx UANL', 'ar CA Huracán', 'co Tolima', 
                'pe Alianza Lima', 'ec CS Emelec', 'pe Universitario', 've Dep Táchira', 
                'bo Oriente Petrolero', 'uy Peñarol', 'uy Defensor', 'uy River Plate', 'bo San José', 
                'ar Boca Juniors', 've Trujillanos', 'pe Juan Aurich', 'ar Rosario Cent', 'bo Bolívar', 
                'bo Wilstermann', 'br Coritiba', 'br Sport Recife', 'bo Universitario', 'ar Banfield', 
                'ar Tigre', 'ar Arsenal', 'mx Toluca', 'ar Belgrano', 've Mineros', 've Zamora', 
                'cl Univ Chile', 'br Cuiabá', 'uy Rentistas', 'co Pereira', 'br Goiás', 
                've Acad Pr Cabello', 'ec SD Aucas', 'cl Ñublense', 'bo Blooming', 'uy Liverpool', 
                've Monagas', 'co La Equidad', 'py Tacuary', 'br América (MG)', 'co Millonarios', 
                'co Independiente', 've Estud Mérida', 'br Internacional', 'uy Boston River', 
                'co AD Cali', 'bo Always Ready', 'pe Melgar', 'cl Antofagasta', 'ar Colón', 
                'py Club General Caballero JLM', 'cl CD Everton', 'br Atl Paranaense', 
                'ec Univ Católica', 'py Cerro Porteño', 've Metropolitanos', 'ar Argentinos Jun', 
                'bo Guabirá', 'uy Torque', 'br Bahia', 've Aragua', 'ar Independiente'
                ]

    teams_to_drop_ita = [
                "at Rapid Wien", 
                "Reggiana", "Parma", 
                "FeralpiSalò", "Spezia",
                "Cosenza", "Sampdoria", 
                "Ternana", "Pisa", "Como", 
                "Ascoli", "Modena", "Cittadella", 
                "Palermo", "Cesena", "Catanzaro"
                ]

    teams_to_drop_spain = [
                "Manchester Utd", "Bayern Munich", "Juventus", "Manchester City", "Chelsea", "Paris S-G", "Liverpool",
                "Leverkusen", "Dortmund", "Roma", "Napoli", "Shakhtar", "Lyon", "PSV Eindhoven", "Milan", "Ajax", "Celtic",
                "Inter", "Porto", "Zenit", "RB Salzburg", "Krasnodar", "Sevilla", "Sporting CP", "Galatasaray", "Arsenal",
                "M'Gladbach", "Viktoria Plzeň", "Benfica", "Club Brugge", "Valencia", "Monaco", "Olympiacos", "Panathinaikos",
                "Loko Moscow", "APOEL FC", "Ludogorets", "Spartak Moscow", "Rapid Wien", "Dinamo Zagreb", "BATE Borisov",
                "Rennes", "Dynamo Kyiv", "Basel", "Leonesa", "Alcoyano", "FC Astana", "CD Mirandés", "Schalke 04", "Rijeka",
                "FC Copenhagen", "AZ Alkmaar", "Ferencváros", "Genk", "Mirandés", "Rubin Kazan", "SD Formentera", "Lazio",
                "Standard Liège", "Slavia Prague", "Hannover 96", "Sparta Prague", "Atalanta", "Lille", "CSKA Moscow", "Lleida",
                "Athletic Club", "RB Leipzig", "Ponferradina", "Eint Frankfurt", "Alcorcón", "Tenerife", "Malmö", "Marseille",
                "Barcelona", "Fuenlabrada", "Rangers", "Young Boys", "Ebro", "Leicester City", "Cornellà", "Numancia", "Wolves",
                "Albacete", "UD Ibiza", "Trabzonspor", "CP Cacereño", "Real Murcia", "CD Arenteiro", "Oviedo", "AD Ceuta",
                "Atlético Baleares", "Gimnàstic", "Zamora CF", "Académica", "Žalgiris", "Sigma Olomouc", "Motherwell",
                "Akhisarspor", "Jablonec", "Hapoel Tel Aviv", "Union Berlin", "St. Gallen", "Kuban", "Swansea City", "Lugo",
                "Újpest", "Twente", "Helsingborg", "Melilla", "Betis", "Braga", "Sant Andreu", "Slaven Belupo", "Kiryat Shmona",
                "Antwerp", "Stoke City", "Strømsgodset", "Udinese", "Beşiktaş", "Luzern", "ŠK Slovan Bratislava", "Stjarnan",
                "Qarabağ FK", "Steaua", "Slovan Liberec", "Reus", "MŠK Žilina", "Inter Baku", "Dinamo Minsk", "Dinamo",
                "Hertha BSC", "Östersund", "Real Jaén", "Partizan", "Legia Warsaw", "Hércules", "Rostov", "Sassuolo", "Guijuelo",
                "Toledo", "Augsburg", "FK Vardar", "Osmanlıspor", "Apollon Limassol", "Estoril", "Śląsk Wrocław", "Mladost",
                "Aberdeen", "Torino", "Başakşehir", "Fiorentina", "Linense", "Sabadell", "L'Hospitalet", "Llagostera",
                "Rosenborg", "Barakaldo", "Gent", "Freiburg", "Odense", "Atlético Mancha Real", "Intercity", "CD Rincón", "APOEL",
                "UD Llanera", "CFJ Mollerussa", "Aris Limassol FC", "Marbella", "Maccabi Haifa", "Real Unión", "Fenerbahçe",
                "Club Portugalete", "Lens", "CF La Nucía", "Navalcarnero", "Sivasspor", "CFR Cluj", "Pontevedra", "West Ham",
                "Bergantiños FC", "Sturm Graz", "Hajduk Split", "Lech Poznań", "Be'er Sheva", "CD Guijuelo", "FC Andorra",
                "SD Leioa", "Badajoz", "Badalona", "Unionistas Sal", "Sestao", "PAOK", "Andratx", "CF Independiente Alicante",
                "Majadahonda", "Guadalajara", "CD L'Alcora", "CF Talavera de la Reina", "Club Deportivo Atlético Paso",
                "CF Panadería Pulido", "Recreativo", "Arenas Club de Getxo", "CD Autol", "CD Cazalegas", "CD Coria",
                "CD Santa Amalia", "Ibiza", "CD Fuentes", "CD Arnedo", "UD Alzira", "CD Eldense", "Quintanar del Rey",
                "UD Barbadás", "Atlético Saguntino", "Velarde CF", "Juventud Torremolinos CF", "Victoria CF", "Gernika Club",
                "CD San Roque de Lepe", "CD Diocesano", "Dnipro", "CD Algar", "Solares SD", "Atlético Sanluqueño CF",
                "CE L'Hospitalet", "CD Cantolagua", "Peña Deportiva", "Racing Rioja CF", "Las Rozas CF", "UM Escobedo",
                "Comillas", "Orihuela CF", "Becerril", "UD Sanse"
            ]   
    
    teams_to_drop_germ = [
                "es Real Madrid", "eng Manchester City", "eng Arsenal", "es Atlético Madrid", "es Barcelona", 
                "fr Paris S-G", "ua Shakhtar","es Sevilla", "pt Benfica", "at RB Salzburg", "eng Tottenham", "gr Olympiacos", 
                "pt Porto", "fr Marseille", "eng Chelsea",    "it Lazio", "it Napoli", "fr Monaco", "it Juventus", "pt Sporting CP", 
                "Holstein Kiel", "it Inter", "eng Manchester Utd",    "ru Zenit", "sct Celtic", "dk FC Copenhagen", "be Club Brugge", 
                "nl Ajax", "eng Liverpool", "it Roma", "es Villarreal",    "tr Galatasaray", "ru CSKA Moscow", "rs Red Star", "gr PAOK", 
                "ru Krasnodar", "Elversberg", "Magdeburg", "sct Rangers",    "St. Pauli", "cz Viktoria Plzeň", "be Anderlecht", "ch Young Boys", 
                "it Atalanta", "Carl Zeiss Jena", "cy APOEL FC", "ro Steaua",    "ru Loko Moscow", "no Molde", "fr Nice", "ch Zürich", "tr Başakşehir", 
                "be Standard Liège", "pt Braga", "ch Basel", "es Athletic Club",    "Sandhausen", "by BATE Borisov", "Karlsruher", "cz Slavia Prague", 
                "Dresden", "be Union SG", "Chemnitzer FC", "tr Beşiktaş",    "Energie Cottbus", "Viktoria Köln", "be Gent", "fr Lille", "Hansa Rostock", 
                "Osnabrück", "Erzgebirge Aue", "1860 Munich", "fr Lyon",    "cz Slovan Liberec", "no Rosenborg", "ua Zorya Luhansk", "fr Saint-Étienne", 
                "nl PSV Eindhoven", "at Wolfsberger AC", "Lotte",    "tr Fenerbahçe", "gr Asteras Tripoli", "az Qabala", "il Maccabi Haifa", "es Real Sociedad", 
                "SV Waldhof", "MSV Duisburg",    "W'burg Kickers", "Jahn R'burg", "bg Ludogorets", "fi HJK", "hu Ferencváros", "be Genk", "se Malmö", "eng West Ham", 
                "Wehen",    "rs Partizan", "Unterhaching", "nl Twente", "Preußen Münster", "Bremer SV", "az Qarabağ Ağdam", "ua Dynamo Kyiv", "es Betis", "se Häcken", 
                "es Valencia", "Saarbrücken", "FSV Frankfurt", "gr AEK Athens", "S'gart Kickers", "Ulm", "Rödinghausen", "Drochtersen/A",    "es Levante", 
                "pl Legia Warsaw", "pl Śląsk Wrocław", "Trier", "ua Vorskla Poltava", "cy Apollon", "ru Rostov", "ie St Patrick's",    
                "ru Dynamo Mosc", "Schweinfurt 05", "si NK Domžale", "se Östersund", "cy AEK Larnaca FC", "ru Anzhi", "Erndtebrück",    
                "se Helsingborg", "az Qarabağ FK", "cy AEL Limassol", "fr Montpellier", "Aalen", "si NK Maribor", "il Maccabi Tel Aviv",   
                "pt Estoril", "cy Apollon Limassol", "ba FK Sarajevo", "eng Everton", "nl AZ Alkmaar", "cz Sparta Prague", "bg Botev Plovdiv",    
                "hr Rijeka", "cy AÉK Lárnaka", "es Málaga", "no Odd", "hr Dinamo Zagreb", "Astoria", "ua Metalist Kharkiv", "at Rapid Wien",   
                "de M'Gladbach", "fr Bordeaux", "de Schalke 04", "it Fiorentina", "dk Brøndby", "FC Teutonia Ottensen", "ro Gaz Metan",    "ee Flora", 
                "fr Strasbourg", "Homburg", "ua FC Oleksandriya", "Hallescher", "fr Nantes", "hu Vidi", "cz Slovácko", "Viktoria 1889",    "Babelsberg 03",
                "ru Spartak Moscow", "fi KuPS", "Rot-Weiss Essen", "nl Feyenoord", "il Be'er Sheva", "ro CS U Craiova",    "be Antwerp", "li Vaduz", "Meppen", 
                "pt Vitória", "Loko Leipzig", "Villingen", "Lübeck", "SV Oberachern", "Verl", "it Milan",    "eng Newcastle Utd", "rs TSC Bačka Top", "Halberstadt", 
                "sct Aberdeen", "bg Levski Sofia", "Paloma", "Greifswalder SV 04",    "Wuppertaler", "TSG Balingen", "Offenbacher K", "de Bayern Munich", "Waldalgesheim",
                "Astoria Walldorf", "Rostocker FC",    "TuS Makkabi Berlin", "Illertissen", "FT Braunschw", "Türkgücü München", "TuS Bersenbrück", "TSV Schott Mainz",
                "Blau-Weiß Lohne",    "de Dortmund", "Hessen Kassel", "Pforzheim", "ua Desna Chernihiv", "Berliner FC Dynamo", "Leher", "LSK Hansa", "Bonner SC",    
                "RW Essen", "Eichstätt", "Dorfmerkingen", "Rielasingen", "Rot-Weiß Erfurt", "BFC Dynamo", "SV Linx", "TSV Steinbach", "SV Atlas",    "RW Koblenz", 
                "Weiche", "Worms", "Hastedt", "Norderstedt", "Aachen", "SpVgg Bayreuth", "Uerdingen", "Union Fürstenwalde",    "al Kukësi", "Nöttingen", 
                "Eintracht Norderstedt", "Zwickau", "Ravensburg", "Hauenstein", "Oberneuland", "Havelse",    "Eintracht Celle", "Babelsberg", "VSG Altglienicke",
                "BFC Preußen", "Germania E/L", "Schweinfurt", "1. FC Düren"
                ]
    
    teams_to_drop_fr = [
                "eng Arsenal", "it Juventus", "nl Ajax", "eng Manchester City", "eng Chelsea", "es Barcelona",
                "de Dortmund", "de Bayern Munich", "de RB Leipzig", "gr Olympiacos", "eng Tottenham", "ua Dynamo Kyiv",
                "at RB Salzburg", "pt Benfica", "de Leverkusen", "es Real Madrid", "pt Porto", "it Lazio", "es Valencia",
                "be Anderlecht", "ch Basel", "rs Red Star", "be Club Brugge", "ua Shakhtar", "de Eint Frankfurt",
                "es Athletic Club", "sct Celtic", "es Sevilla", "cz Sparta Prague", "ru Zenit", "Red Star", "Tours",
                "it Napoli", "ru CSKA Moscow", "pt Braga", "it Inter", "cy APOEL FC", "de Wolfsburg", "il Maccabi Haifa",
                "es Villarreal", "hr Dinamo Zagreb", "eng Liverpool", "eng Manchester Utd", "nl PSV Eindhoven",
                "tr Fenerbahçe", "ru Krasnodar", "be Gent", "Bourg-Péronnas", "Châteauroux", "Grenoble",
                "es Atlético Madrid", "Paris FC", "md FC Milsami Orhei", "no Rosenborg", "ch Grasshopper", "nl Vitesse",
                "Épinal", "ru Rubin Kazan", "ua Dnipro", "dk FC Copenhagen", "az Qarabağ FK", "pt Vitória", "az Qabala",
                "Granville", "tr Beşiktaş", "de Schalke 04", "Le Mans", "es Betis", "cy Apollon Limassol",
                "eng Everton", "cz Slavia Prague", "az Qarabağ Ağdam", "Dunkerque", "hr Rijeka", "it Milan",
                "ru Loko Moscow", "il Maccabi Tel Aviv", "tr Galatasaray", "Niort", "eng Newcastle Utd", "Pau FC",
                "md Sheriff Tiraspol", "tr Trabzonspor", "gr Panathinaikos", "es Real Sociedad", "gr PAOK",
                "Rodez Aveyron", "gr AEK Athens", "Bergerac", "Laval", "Étoile FC FSR", "Chambly Oise", "St-Pryvé St-H.",
                "Linas-Montl.", "Bastia-Borgo", "Les Herbiers", "Avranches MSM", "Orléans", "it Atalanta", "Trélissac",
                "be Union SG", "it Roma", "il Beitar Jerusalem", "de Mainz 05", "at LASK", "nl AZ Alkmaar",
                "Quevilly-Rouen", "bg Ludogorets", "be Zulte Waregem", "hu Videoton", "pt Sporting CP", "tr Konyaspor",
                "si NK Domžale", "be Oostende", "Annecy", "cy AÉK Lárnaka", "Yzeure", "hu Ferencváros", "IC Croix",
                "ua FC Mariupol", "lv FK Ventspils", "sk Slovan Bratislava", "ro Târgu Mureș", "Besançon", "de M'Gladbach",
                "Arles-Avignon", "dk Esbjerg", "ua Chornomorets", "cz Viktoria Plzeň", "tr Eskişehirspor",
                "cy AEL Limassol", "il Kiryat Shmona", "it Fiorentina", "by BATE Borisov", "pt Marítimo", "lu Differdange 03",
                "sk ŠK Slovan Bratislava", "ge FC Rustavi", "it Udinese", "by Dinamo Minsk", "Créteil", "se Malmö",
                "eng Brighton", "ch Young Boys", "fo KÍ", "si Olimpija", "FCO Strasbourg Koenigshoffen 06", "cy AEK Larnaca FC",
                "kz FC Kairat", "ch Sion", "tr Karabükspor", "nl Groningen", "cz Slovan Liberec", "US Boulogne",
                "US Concarneau", "cz Mladá Boleslav", "ro Astra Giurgiu", "Viry-Châtillon", "Canet RFC", "kz Astana FK",
                "bg Loko Plovdiv", "Bourg Péronnas", "Le Puy Foot 43", "US Raon", "ASM Belfort", "ro CFR Cluj",
                "C. Franciscain", "Be'er Sheva", "Perpignan Canet FC", "Başakşehir", "de Freiburg", "Thaon", "EF Reims",
                "Dinan-Léhon", "Jura Sud Foot", "eng West Ham", "dk Brøndby", "sct Rangers", "eng Leicester City",
                "si NŠ Mura", "nl Feyenoord", "at Sturm Graz", "cz Slovácko", "ua Metalist Kharkiv", "de Köln",
                "cz Jablonec", "Entente SSG", "de Hoffenheim", "FC Rouen", "ua FC Oleksandriya", "cy Apollon", "rs Partizan",
                "Saint-Malo", "JS Chemin Bas d'Avignon", "AS Lyon-Duchère", "AS Jumeaux de Mzouazia", "Bourges", "Vire",
                "Dieppois", "Sainte-Marien.", "Pontivy", "Chantilly", "Sarre-Union", "Mantois", "Blanc-Mesnil", "Sète",
                "ASF Andrézieux", "Vannes", "SC Bastia", "Andrézieux", "Lyon-Duchère", "SO Chambéry Foot", "SO Cholet"
            ]

    teams_to_drop_gr = [
                "nl Ajax", "sk Slovan Bratislava", "fr Marseille", "hr Rijeka", "ru Krasnodar", "eng Arsenal",
                "de Bayern Munich", "az Qabala", "ua Dynamo Kyiv", "pt Benfica", "cy APOEL FC", "il Maccabi Tel Aviv",
                "cy AC Omonia", "nl PSV Eindhoven", "be Club Brugge", "it Milan", "tr Fenerbahçe", "hu Vidi",
                "de Freiburg", "se AIK Stockholm", "il Maccabi Haifa", "pt Braga", "de Eintracht Frankfurt",
                "hr Dinamo Zagreb", "be Antwerp", "de Schalke 04", "dk Brøndby", "tr Beşiktaş", "cy Apollon",
                "sct Celtic", "de Dortmund", "nl AZ Alkmaar", "ch Luzern", "eng Burnley", "at Austria Wien",
                "es Betis", "be Anderlecht", "ru CSKA Moscow", "by FC Dinamo Brest", "hr Lokomotiva",
                "lu F91 Dudelange", "sk Spartak Trnava", "fr Saint-Étienne", "sct Hibernian", "no Odd",
                "it Fiorentina", "ua Olimpik Donetsk", "il Be'er Sheva", "ge Dinamo Tbilisi", "tr Osmanlıspor",
                "cz Slovan Liberec", "az Qarabağ FK", "kz FC Astana", "ch Young Boys", "pt Arouca", "es Athletic Club",
                "se Östersund", "es Celta Vigo", "si ND Gorica", "es Barcelona", "be Standard Liège", "it Juventus",
                "by BATE Borisov", "rs Partizan", "pt Sporting CP", "be Genk", "eng Chelsea", "eng Brighton",
                "ie Bohemian FC", "it Atalanta", "bg Ludogorets", "az Neftçi", "by FC Gomel", "bg Levski Sofia",
                "az Qarabağ Ağdam", "fr Nantes", "cz Slavia Prague", "am Ararat-Armenia", "sct Aberdeen",
                "dk FC Copenhagen", "fi HJK", "sct Hearts", "hr Hajduk Split", "il Beitar Jerusalem", "fr Rennes",
                "es Villarreal", "ua SK Dnipro-1", "eng West Ham", "rs TSC Bačka Top", "rs Čukarički", "gi Red Imps",
                "dk Midtjylland", "ru Spartak Moscow", "tr Başakşehir", "ch Basel", "pl Legia Warsaw",
                "sk Dunajská Streda", "no Molde", "cy AE Lemesós", "tr Trabzonspor", "ro CS U Craiova", "eng Wolves",
                "rs Red Star", "eng Tottenham", "cz Viktoria Plzeň", "be Gent", "es Granada", "ua Zorya Luhansk",
                "eng Leicester City", "eng Manchester City", "pt Porto", "kz Astana FK", "ba FK Velež Mostar",
                "cz Sparta Prague", "de Wolfsburg", "ch St. Gallen", "ua FK Kolos"
            ]   


    if url == url_arg:        
        df = df[~df['opponent'].isin(teams_to_drop_arg)]
    elif url == url_ita:
        df = df[~df['opponent'].isin(teams_to_drop_ita)]
    elif url == url_spain:
        df = df[~df['opponent'].isin(teams_to_drop_spain)]
    elif url == url_germ:
        df = df[~df['opponent'].isin(teams_to_drop_germ)]
    elif url == url_fr:
        df = df[~df['opponent'].isin(teams_to_drop_fr)]
    elif url == url_gr:
        df = df[~df['opponent'].isin(teams_to_drop_gr)]  

    return df

def export_model(model):
    export = pickle.dump(model, open('../rf.pkl', 'wb'))
    return export

def export_model_v2(model, name):
    """
    A function to export the models trained with the different datasets.
    Parameters used:
        - model : the trained model
        - name : a variable to identify each model. It is used inside the main function,
                 and it can be assigned inside of each if statement with the corresponding name. 
                 e.g.   name = ""
                        if idx == 1:
                            name = "brazil_serie_a"
    """
    export = pickle.dump(model, open('../rf_' + name +'.pkl', 'wb'))
    return export

if __name__ == "__main__":

    url_br_a = "../Datasets/Cleaned Datasets/Serie_A_Stats.csv"
    url_br_b = "../Datasets/Cleaned Datasets/Serie_B_Stats.csv"
    url_arg = "../Datasets/Cleaned Datasets/Primera_Division_Stats.csv"
    url_jpn = "../Datasets/Cleaned Datasets/J1_League_Stats.csv"
    url_nor = "../Datasets/Cleaned Datasets/Eliteserien_Stats.csv"
    url_fin = "../Datasets/Cleaned Datasets/Veikkausliiga_Stats.csv"
    url_eng = "../Datasets/Cleaned Datasets/Premier_League_Stats.csv"
    url_ita = "../Datasets/Cleaned Datasets/Serie_A_Stats_Italy.csv"
    url_spain = "../Datasets/Cleaned Datasets/La_Liga_Stats.csv"
    url_germ = "../Datasets/Cleaned Datasets/Bundesliga_Stats.csv"
    url_fr = "../Datasets/Cleaned Datasets/Ligue_1_Stats.csv"
    url_gr = "../Datasets/Cleaned Datasets/Super_League_Greece_Stats.csv"

    # urls = [url_br_a, url_br_b, url_arg, url_jpn, url_nor, url_fin]

    urls = [url_br_a, url_br_b, url_arg, 
            url_jpn, url_nor, url_fin, 
            url_eng, url_ita, url_spain,
            url_germ, url_fr, url_gr]
    
    for idx, url in enumerate(urls):
        
        df_train = read_train(url)
        
        if url == url_arg or url == url_ita or url == url_germ or url == url_fr or url == url_gr:
            df_train = drop_teams(df_train, url)

        df_train = create_predictors(df_train)
                
        cols = ["gf", "ga", "sh", "sot", "pk", "pkatt"]
        new_cols = [f"{c}_rolling" for c in cols]
        matches_rolling = update_with_rolling_average(df_train, cols, new_cols)
        
        predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
        rf_model, combined = train_model(matches_rolling, predictors + new_cols, url)

        combined_df = combine(combined, matches_rolling)
        # combined_df = combined_df.drop(['actual', 'result'], axis = 1)
    
        final_df = map_values(combined_df)

        # Export the DataFrame to a CSV file
        if idx == 0:
            final_df.to_csv('../Datasets/Predictions/predictions_serieA.csv', index=False)
        elif idx == 1:
            final_df.to_csv('../Datasets/Predictions/predictions_serieB.csv', index=False)
        elif idx == 2:
            final_df.to_csv('../Datasets/Predictions/predictions_primera_division.csv', index=False)
        elif idx == 3:
            final_df.to_csv('../Datasets/Predictions/predictions_j1_league.csv', index=False)
        elif idx == 4:
            final_df.to_csv('../Datasets/Predictions/predictions_eliteserien_nor_league.csv', index=False)
        elif idx == 5:
            final_df.to_csv('../Datasets/Predictions/predictions_veikkausliiga_fin_league.csv', index=False)
        elif idx == 6:
            final_df.to_csv('../Datasets/Predictions/predictions_premier_league_england.csv', index=False)
        elif idx == 7:
            final_df.to_csv('../Datasets/Predictions/predictions_serieA_italy.csv', index=False)
        elif idx == 8:
            final_df.to_csv('../Datasets/Predictions/predictions_laliga_spain.csv', index=False)
        elif idx == 9:
            final_df.to_csv('../Datasets/Predictions/predictions_bundesliga_germany.csv', index=False)
        elif idx == 10:
            final_df.to_csv('../Datasets/Predictions/predictions_ligue1_france.csv', index=False)
        elif idx == 11:
            final_df.to_csv('../Datasets/Predictions/predictions_super_league_greece.csv', index=False)


        export_model(rf_model)


