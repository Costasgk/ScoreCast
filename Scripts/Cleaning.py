import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from datetime import datetime, date


def read_file(path):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # FBref tables embed repeated header rows in tbody — drop them
    if 'date' in df.columns:
        df = df[df['date'].astype(str).str.lower() != 'date']

    if 'sh.1' in df.columns:
        df = df.drop(['sh.1'], axis = 1)

    if ('np:g-xg' in df.columns or 'npxg' in df.columns or 'npxg/sh' in df.columns or
        'xg' in df.columns or 'xga' in df.columns or 'g-xg' in df.columns):
        df_keep = df[['year', 'date', 'time', 'day', 'comp', 'round', 'venue', 'referee','team', 'opponent', 
                    'formation', 'result', 'poss',  'gf', 'ga', 
                    'gls', 'sh', 'sot', 'sot%', 'g/sh', 
                    'g/sot', 'pk', 'pkatt', 'g-xg', 'xg', 'xga',
                    'np:g-xg', 'npxg', 'npxg/sh' ]]
    else:
        df_keep = df[['year', 'date', 'time', 'day', 'comp', 'round', 'venue', 'referee','team', 'opponent', 
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

    for col in ['poss', 'gf', 'ga', 'gls', 'sh', 'sot', 'sot%', 'g/sh', 'g/sot', 'pk', 'pkatt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if df['poss'].isnull().all():
        df = df.drop(['poss'], axis = 1)
    else:
        df['poss'] = df['poss'].fillna(df['poss'].mean())

    # gf/ga left as NaN where missing — fake mean-filled goals corrupt the model

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

def export_df(path, df, out_name=None):
    if out_name is None:
        out_name = path.split('/')[3]
    out_path = '../Datasets/Cleaned Datasets/' + out_name
    df.to_csv(out_path)
    years = sorted(df['year'].dropna().unique().astype(int)) if 'year' in df.columns else []
    year_range = f"{years[0]}-{years[-1]}" if years else "unknown"
    print(f"  Exported: {out_name:<40} {len(df):>6} rows  |  years {year_range}")
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
    # (scrapped input, cleaned output name)
    leagues = [
        ('../Datasets/Scrapped Datasets/Serie_A_Stats.csv',             'Brazil_Serie_A.csv'),
        ('../Datasets/Scrapped Datasets/Eliteserien_Stats.csv',         'Norway_Eliteserien.csv'),
        ('../Datasets/Scrapped Datasets/Veikkausliiga_Stats.csv',       'Finland_Veikkausliiga.csv'),
        ('../Datasets/Scrapped Datasets/Super_League_Greece_Stats.csv', 'Greece_Super_League.csv'),
        ('../Datasets/Scrapped Datasets/Premier_League_Stats.csv',      'England_Premier_League.csv'),
        ('../Datasets/Scrapped Datasets/Serie_A_Stats_Italy.csv',       'Italy_Serie_A.csv'),
        ('../Datasets/Scrapped Datasets/La_Liga_Stats.csv',             'Spain_La_Liga.csv'),
        ('../Datasets/Scrapped Datasets/Bundesliga_Stats.csv',          'Germany_Bundesliga.csv'),
        ('../Datasets/Scrapped Datasets/Ligue_1_Stats.csv',             'France_Ligue_1.csv'),
    ]

    print()
    print('  ScoreCast Cleaner  |  10 leagues')
    print()
    for path, out_name in leagues:
        df = read_file(path)
        df = cleaning(df)
        export_df(path, df, out_name)
    print()

if __name__ == "__main__":
    process()
