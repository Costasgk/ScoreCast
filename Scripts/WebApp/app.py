from flask import Flask, render_template, request
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

    df = df[(df['Date'].dt.date < pd.to_datetime('2023-08-31').date()) & (df['Date'].dt.date >= pd.to_datetime('2023-07-30').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('br_a.html', table_html=table_html)

@app.route('/Serie-B-Brazil', methods = ['GET'])
def display_brazil_B():
    # csv_file = '../../Datasets/Predictions/predictions_serieB.csv'
    csv_file = get_predictions_file_path('predictions_serieB.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'].dt.date < pd.to_datetime('2023-08-31').date()) & (df['Date'].dt.date >= pd.to_datetime('2023-07-30').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('br_b.html', table_html=table_html)

@app.route('/Primera-Division-Argentina', methods = ['GET'])
def display_argentina():
    # csv_file = '../../Datasets/Predictions/predictions_premiera_division.csv'
    csv_file = get_predictions_file_path('predictions_premiera_division.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[df['Date'].dt.date < pd.to_datetime('2023-08-31').date()]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('arg.html', table_html=table_html)

@app.route('/J1-League-Japan', methods = ['GET'])
def display_japan():
    # csv_file = '../../Datasets/Predictions/predictions_j1_league.csv'
    csv_file = get_predictions_file_path('predictions_j1_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'].dt.date < pd.to_datetime('2023-08-31').date()) & (df['Date'].dt.date >= pd.to_datetime('2023-07-30').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('jpn.html', table_html=table_html)

@app.route('/Eliteserien-Norway', methods = ['GET'])
def display_norway():
    # csv_file = '../../Datasets/Predictions/predictions_eliteserien_nor_league.csv'

    csv_file = get_predictions_file_path('predictions_eliteserien_nor_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'].dt.date < pd.to_datetime('2023-08-31').date()) & (df['Date'].dt.date >= pd.to_datetime('2023-07-30').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('norw.html', table_html=table_html)

@app.route('/Veikkausliiga-Finland', methods = ['GET'])
def display_finland():
    # csv_file = '../../Datasets/Predictions/predictions_veikkausliiga_fin_league.csv'
    csv_file = get_predictions_file_path('predictions_veikkausliiga_fin_league.csv')

    df = pd.read_csv(csv_file)
    
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'].dt.date < pd.to_datetime('2023-08-31').date()) & (df['Date'].dt.date >= pd.to_datetime('2023-07-30').date())]
 
    table_html = df.to_html(index=False, classes='table table-bordered table-striped')

    return render_template('fin.html', table_html=table_html)


if __name__ == '__main__':
    app.run(debug=True)

