import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.request
import os
import time

def scrape_data():

    J1_jpn = "https://fbref.com/en/comps/25/J1-League-Stats"
    # SerieA_argentina = "https://fbref.com/en/comps/21/Primera-Division-Stats"
    SerieA_brazil = "https://fbref.com/en/comps/24/Serie-A-Stats"
    SerieB_brazil = "https://fbref.com/en/comps/38/Serie-B-Stats"
    LeagueA_norway = "https://fbref.com/en/comps/28/Eliteserien-Stats"
    LeagueA_finland = "https://fbref.com/en/comps/43/Veikkausliiga-Stats"
    PL_england = "https://fbref.com/en/comps/9/Premier-League-Stats"
    SerieA_italy = "https://fbref.com/en/comps/11/Serie-A-Stats"
    LaLiga_spain = "https://fbref.com/en/comps/12/La-Liga-Stats"
    LigueA_france = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    Bundesliga_germany = "https://fbref.com/en/comps/20/Bundesliga-Stats"
    SuperLeague_greece = "https://fbref.com/en/comps/27/Super-League-Greece-Stats"

    urls = [SerieA_brazil, SerieB_brazil, 
            LeagueA_norway, LeagueA_finland,  
            SuperLeague_greece, PL_england, 
            SerieA_italy, LaLiga_spain, 
            Bundesliga_germany, LigueA_france, J1_jpn ]
    
    for url in urls:

        standings_url = url
        splitted_url = standings_url.split('/')
        last_split = splitted_url[-1]
        
        if url == SerieB_brazil:
            years = list(range(2023,2019,-1))
        elif url == PL_england or url == SerieA_italy or url == LaLiga_spain or url == LigueA_france or url == Bundesliga_germany:
            years = list(range(2023, 2010, -1))
        else:
            years = list(range(2023,2014,-1))

        if url == SerieA_italy:
            last_split = last_split + "-Italy"
        
        all_matches = []

        for year in years:

            print("\n Scrapping Data for Year", year)
            
            data = requests.get(standings_url)
            # soup = BeautifulSoup(data.text)
            soup = BeautifulSoup(data.text, "html.parser")

            standings_table = soup.select("table.stats_table")[0]

            links = standings_table.find_all('a')

            href = [link.get('href') for link in links]
            href_squad = [link for link in href if '/squads/' in link]
            team_urls = [f'https://fbref.com{link}' for link in href_squad]

            try:
                previous_season = soup.select("a.prev")[0].get("href")
                standings_url = f'https://fbref.com{previous_season}'
            except IndexError:
                previous_season = ""
                standings_url = ""

            for team_url in team_urls:
                team_name = team_url.split('/')[-1].replace('-Stats', '').replace('-', ' ')

                data = requests.get(team_url)
                matches = pd.read_html(data.text, match = 'Scores & Fixtures')[0]

                soup = BeautifulSoup(data.text, "html.parser")
                links = soup.find_all('a')
                href = [link.get('href') for link in links]
                href_shooting = [link for link in href if link and 'all_comps/shooting' in link]
                data = requests.get(f'https://fbref.com/{href_shooting[0]}')

                shooting = pd.read_html(data.text, match = 'Shooting')[0]
                shooting.columns = shooting.columns.droplevel()

                try:
                    team_data = pd.concat([matches, shooting[shooting.columns.difference(matches.columns)]], axis=1)
                except ValueError:
                    print("A column in missing")
                    continue
                
                print("\n > Team Appending:", team_name)
                
                team_data['Year'] = year
                team_data['Team'] = team_name

                all_matches.append(team_data)
                time.sleep(10)

        match_df = pd.concat(all_matches)

        match_df.columns = [c.lower() for c in match_df.columns]
        
        last_split = last_split.replace('-', '_')

        match_df.to_csv("../Datasets/Scrapped Datasets/" + last_split + ".csv")

if __name__ == "__main__":
    scrape_data()