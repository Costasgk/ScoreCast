import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from datetime import datetime, date


def read_file(path):
    df = pd.read_csv(path)

    if 'sh.1' in df.columns:
        df = df.drop(['sh.1'], axis = 1)

    if ('np:g-xg' in df.columns or 'npxg' in df.columns or 'npxg/sh' in df.columns or
        'xg' in df.columns or 'xga' in df.columns or 'g-xg' in df.columns):
        df_keep = df[['year', 'date', 'time', 'day', 'venue', 'referee','team', 'opponent', 
                    'formation', 'result', 'poss',  'gf', 'ga', 
                    'gls', 'sh', 'sot', 'sot%', 'g/sh', 
                    'g/sot', 'pk', 'pkatt', 'g-xg', 'xg', 'xga',
                    'np:g-xg', 'npxg', 'npxg/sh' ]]
    else:
        df_keep = df[['year', 'date', 'time', 'day', 'venue', 'referee','team', 'opponent', 
                    'formation', 'result', 'poss',  'gf', 'ga', 
                    'gls', 'sh', 'sot', 'sot%', 'g/sh', 
                    'g/sot', 'pk', 'pkatt' ]]
    
    return df_keep

def split_dataset(df):
    current_date = date.today()
    df['date'] = pd.to_datetime(df['date'])
    # df['day'] = df
    # df_filtered = df.sort_values(by='date', ascending=False)

    df_train = df[df['date'] <= current_date]
    df_test = df[df['date'] > current_date]

    return df_train, df_test

def cleaning(df):

    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['time'].astype('str')

    if df['venue'].isnull().any():
        df['venue'] = df['venue'].fillna(df['venue'].value_counts().idxmax())

    if df['opponent'].isnull().any():
        df['opponent'] = df['opponent'].fillna(df['opponent'].value_counts().idxmax())

    if df['formation'].isnull().all():
        df = df.drop(['formation'], axis = 1)
    else:
        df['formation'] = df['formation'].fillna(df['formation'].value_counts().idxmax())
    
    if df['result'].isnull().any():
        df['result'] = df['result'].fillna(df['result'].value_counts().idxmax())

    if df['poss'].isnull().all():
        df = df.drop(['poss'], axis = 1)
    else:
        df['poss'] = df['poss'].fillna(df['poss'].mean())
    
    if df['gf'].isnull().any() & df['ga'].isnull().any():
        df['gf'] = pd.to_numeric(df['gf'], errors='coerce')
        df['ga'] = pd.to_numeric(df['gf'], errors='coerce')
        df['gf'] = df['gf'].fillna(df['gf'].mean())
        df['ga'] = df['ga'].fillna(df['ga'].mean())

    df['referee'] = df['referee'].fillna(df['referee'].value_counts().idxmax())
    df['gls'] = df['gls'].fillna(df['gls'].mean())
    df['sh'] = df['sh'].fillna(df['sh'].mean())
    df['sot'] = df['sot'].fillna(df['sot'].mean())
    df['sot%'] = df['sot%'].fillna(df['sot%'].mean())
    df['g/sh'] = df['g/sh'].fillna(df['g/sh'].mean())
    df['g/sot'] = df['g/sot'].fillna(df['g/sot'].mean())
    df['pk'] = df['pk'].fillna(df['pk'].mean())
    df['pkatt'] = df['pkatt'].fillna(df['pkatt'].mean())    

    # if ('g-xg' or 'xg' or 'xga' or 'np:g-xg' or 'npxg' or 'npxg/sh') in df.columns:
    #     columns_with_missing = df.columns[df .isnull().any()].tolist()
    #     imputer = IterativeImputer(max_iter=10, random_state=0)  
    #     df[columns_with_missing] = imputer.fit_transform(df[columns_with_missing])


    return df

def export_df(path, df):
    filename = path.split('/')[3]
    export = df.to_csv('../Datasets/Cleaned Datasets/' + filename)
    print('CSV exported >', filename)
    return 'CSV exported'

def export_df_test(path,df):
    filename = path.split('/')[3]
    if filename == 'J1_League_Stats.csv':
        export = df.to_csv('../Datasets/Cleaned Datasets/Testing_' + filename )
        print(f'CSV for testing exported ---> Testing_{filename}')
    else:
        export = df.to_csv('../Datasets/Cleaned Datasets/Testing-' + filename )
        print(f'CSV for testing exported ---> Testing-{filename}')

    return 'CSV Testing exported'


def process():
    path_jpn = '../Datasets/Scrapped Datasets/J1_League_Stats.csv'
    # path_agentinaA = '../Datasets/Scrapped Datasets/Primera-Division-Stats.csv'
    path_brazilA = '../Datasets/Scrapped Datasets/Serie_A_Stats.csv'
    path_brazilB = '../Datasets/Scrapped Datasets/Serie_B_Stats.csv'
    path_norwayA = '../Datasets/Scrapped Datasets/Eliteserien_Stats.csv'
    path_finlandA = '../Datasets/Scrapped Datasets/Veikkausliiga_Stats.csv'
    path_englandPL = '../Datasets/Scrapped Datasets/Premier_League_Stats.csv'
    path_italyA = '../Datasets/Scrapped Datasets/Serie_A_Stats_Italy.csv'

    # paths = [path_jpn, path_agentinaA, path_brazilA, path_brazilB, path_norwayA, path_finlandA, path_englandPL, path_italyA]
    paths = [path_jpn, path_brazilA, path_brazilB, path_norwayA, path_finlandA, path_englandPL, path_italyA]

    for path in paths:
        df = read_file(path)
        # df, df_test = split_dataset(df)
        df = cleaning(df)
        # df = label_encoding(df)
        df = export_df(path,df)

        # df_test = label_encoding(df_test)
        # df_test = export_df_test(path, df_test)

if __name__ == "__main__":
    process()
