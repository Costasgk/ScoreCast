from flask import Flask, render_template, request, send_file
import pickle, re, os
import pandas as pd
from pathlib import Path

app = Flask(__name__)

def get_predictions_file_path(file_name):
    if 'DYNO' in os.environ:
        # Running on Heroku
        base_dir = Path.cwd() 
    else:
        # Running locally
        base_dir = Path(__file__).resolve().parent.parent.parent 

    predictions_dir = base_dir / 'Datasets' / 'Predictions'
    return predictions_dir / file_name

@app.route('/', methods = ['GET'])
def display_home():
    return render_template('index.html')


@app.route('/Serie-A-Brazil', methods = ['GET'])
def display_brazil_A():
    # csv_file = '../../Datasets/Predictions/predictions_serieA.csv'
    csv_file = get_predictions_file_path('predictions_serieA.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('br_a.html', table_html=table_html)

@app.route('/Serie-A-Brazil-csv', methods=['GET'])
def download_csv_brazil_a():
    csv_file = get_predictions_file_path('predictions_serieA.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_serieA.csv')

@app.route('/Serie-B-Brazil', methods = ['GET'])
def display_brazil_B():
    # csv_file = '../../Datasets/Predictions/predictions_serieB.csv'
    csv_file = get_predictions_file_path('predictions_serieB.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('br_b.html', table_html=table_html)

@app.route('/Serie-B-Brazil-csv', methods=['GET'])
def download_csv_brazil_b():
    csv_file = get_predictions_file_path('predictions_serieB.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_serieB.csv')

@app.route('/Primera-Division-Argentina', methods = ['GET'])
def display_argentina():
    # csv_file = '../../Datasets/Predictions/predictions_primera_division.csv'
    csv_file = get_predictions_file_path('predictions_primera_division.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('')

    df = df[df['Date'].dt.date < pd.to_datetime('2023-08-31').date()]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('arg.html', table_html=table_html)

@app.route('/Primera-Division-Argentina-csv', methods=['GET'])
def download_csv_arg():
    csv_file = get_predictions_file_path('predictions_primera_division.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_primera_division.csv')

@app.route('/J1-League-Japan', methods = ['GET'])
def display_japan():
    # csv_file = '../../Datasets/Predictions/predictions_j1_league.csv'
    csv_file = get_predictions_file_path('predictions_j1_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('jpn.html', table_html=table_html)

@app.route('/J1-League-Japan-csv', methods=['GET'])
def download_csv_jpn():
    csv_file = get_predictions_file_path('predictions_j1_league.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_j1_league.csv')

@app.route('/Eliteserien-Norway', methods = ['GET'])
def display_norway():
    # csv_file = '../../Datasets/Predictions/predictions_eliteserien_nor_league.csv'

    csv_file = get_predictions_file_path('predictions_eliteserien_nor_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('norw.html', table_html=table_html)

@app.route('/Eliteserien-Norway-csv', methods=['GET'])
def download_csv_norw():
    csv_file = get_predictions_file_path('predictions_eliteserien_nor_league.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_eliteserien_nor_league.csv')
    
@app.route('/Veikkausliiga-Finland', methods = ['GET'])
def display_finland():
    # csv_file = '../../Datasets/Predictions/predictions_veikkausliiga_fin_league.csv'
    csv_file = get_predictions_file_path('predictions_veikkausliiga_fin_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-01').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('fin.html', table_html=table_html)

@app.route('/Veikkausliiga-Finland-csv', methods=['GET'])
def download_csv_fin():
    csv_file = get_predictions_file_path('predictions_veikkausliiga_fin_league.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_veikkausliiga_fin_league.csv')

@app.route('/Premier-League-England', methods = ['GET'])
def display_england():
    csv_file = get_predictions_file_path('predictions_premier_league_england.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('eng.html', table_html=table_html)

@app.route('/Premier-League-England-csv', methods=['GET'])
def download_csv_eng():
    csv_file = get_predictions_file_path('predictions_premier_league_england.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_premier_league_england.csv')

@app.route('/Serie-A-Italy', methods = ['GET'])
def display_italy():
    csv_file = get_predictions_file_path('predictions_serieA_italy.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('ita.html', table_html=table_html)

@app.route('/Serie-A-Italy-csv', methods=['GET'])
def download_csv_ita():
    csv_file = get_predictions_file_path('predictions_serieA_italy.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_serieA_italy.csv')

@app.route('/La-Liga-Spain', methods = ['GET'])
def display_spain():
    csv_file = get_predictions_file_path('predictions_laliga_spain.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('spa.html', table_html=table_html)

@app.route('/La-Liga-Spain-csv', methods=['GET'])
def download_csv_spa():
    csv_file = get_predictions_file_path('predictions_laliga_spain.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_laliga_spain.csv')

@app.route('/Ligue-1-France', methods=['GET'])
def display_france():
    csv_file = get_predictions_file_path('predictions_ligue1_france.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('fr.html', table_html=table_html)

@app.route('/Ligue-1-France-csv', methods=['GET'])
def download_csv_fr():
    csv_file = get_predictions_file_path('predictions_ligue1_france.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_ligue1_france.csv')

@app.route('/Bundesliga-Germany', methods=['GET'])
def display_germany():
    csv_file = get_predictions_file_path('predictions_bundesliga_germany.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('germ.html', table_html=table_html)

@app.route('/Bundesliga-Germany-csv', methods=['GET'])
def download_csv_germ():
    csv_file = get_predictions_file_path('predictions_bundesliga_germany.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_bundesliga_germany.csv')

@app.route('/Super-League-Greece', methods=['GET'])
def display_greece():
    csv_file = get_predictions_file_path('predictions_super_league_greece.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df['Time'] = df['Time'].fillna('Not announced yet')

    df = df[(df['Date'].dt.date >= pd.to_datetime('2023-09-05').date()) & (df['Date'].dt.date < pd.to_datetime('2023-12-31').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('gr.html', table_html=table_html)

@app.route('/Super-League-Greece-csv', methods=['GET'])
def download_csv_gr():
    csv_file = get_predictions_file_path('predictions_super_league_greece.csv')
    return send_file(csv_file, as_attachment=True, download_name='predictions_super_league_greece.csv')

if __name__ == '__main__':
    app.run(debug=True)

