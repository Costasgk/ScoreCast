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
    combined_data = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']],
                              left_index = True, right_index = True)
    return combined_data

def train_model(data, predictors):
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
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
        
        print(combined_df)

        merged = combined_df.merge(combined_df, left_on=["date", "new_team"], right_on=["date", "opponent"])

        merged = merged[['date', 'new_team_x', 'new_team_y', 'predicted_x', 'predicted_y']]

        map_string = {0: 'L', 1 : 'W', 2 : 'D'}
        
        merged['new_predicted_x'] = merged['predicted_x'].map(map_string)
        merged['new_predicted_y'] = merged['predicted_y'].map(map_string)
        
        renamed_columns = {
            'date' : 'Date',
            'new_team_x' : 'Team A',
            'new_team_y' : 'Team B',
            'new_predicted_x' : 'Prediction for Team A',
            'new_predicted_y' : 'Prediction for Team B'
        }

        merged.rename(columns=renamed_columns, inplace=True)
        
        merged = merged.drop(['predicted_x', 'predicted_y'], axis = 1)

    return merged

def drop_teams(df):

    # Premiera Division Argentina
    teams_to_drop = [
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
            
    df = df[~df['opponent'].isin(teams_to_drop)]

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

    url_br_a = "../Datasets/Cleaned Datasets/Serie-A-Stats.csv"
    url_br_b = "../Datasets/Cleaned Datasets/Serie-B-Stats.csv"
    url_arg = "../Datasets/Cleaned Datasets/Primera-Division-Stats.csv"
    url_jpn = "../Datasets/Cleaned Datasets/J1_League_Stats.csv"
    url_nor = "../Datasets/Cleaned Datasets/Eliteserien-Stats.csv"
    url_fin = "../Datasets/Cleaned Datasets/Veikkausliiga-Stats.csv"
    
    urls = [url_br_a, url_br_b, url_arg, url_jpn, url_nor, url_fin]
    # urls = [url_br_a, url_br_b, url_arg, url_jpn, url_nor, url_fin]
    
    for idx, url in enumerate(urls):
        
        df_train = read_train(url)
        
        if url == url_arg:
            df_train = drop_teams(df_train)
            
        # print(df_train['date'].dt.dayofweek)
        df_train = create_predictors(df_train)
        
        cols = ["gf", "ga", "sh", "sot", "pk", "pkatt"]
        new_cols = [f"{c}_rolling" for c in cols]
        matches_rolling = update_with_rolling_average(df_train, cols, new_cols)
        
        predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
        rf_model, combined = train_model(matches_rolling, predictors + new_cols)

        combined_df = combine(combined, matches_rolling)
        # combined_df = combined_df.drop(['actual', 'result'], axis = 1)
    
        final_df = map_values(combined_df)

        # Export the DataFrame to a CSV file
        if idx == 0:
            final_df.to_csv('../Datasets/Predictions/predictions_serieA.csv', index=False)
        elif idx == 1:
            final_df.to_csv('../Datasets/Predictions/predictions_serieB.csv', index=False)
        elif idx == 2:
            final_df.to_csv('../Datasets/Predictions/predictions_premiera_division.csv', index=False)
        elif idx == 3:
            final_df.to_csv('../Datasets/Predictions/predictions_j1_league.csv', index=False)
        elif idx == 4:
            final_df.to_csv('../Datasets/Predictions/predictions_eliteserien_nor_league.csv', index=False)
        elif idx == 5:
            final_df.to_csv('../Datasets/Predictions/predictions_veikkausliiga_fin_league.csv', index=False)

        export_model(rf_model)


