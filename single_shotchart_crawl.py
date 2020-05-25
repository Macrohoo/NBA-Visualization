import requests
import pandas as pd
import numpy

def getshotchart(url):
    try:
        kv = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:75.0) Gecko/20100101 Firefox/75.0",
            "Referer": "https://stats.nba.com/events/?flag=3&CFID=33&CFPARAMS=2019-20&PlayerID=201935&ContextMeasure=FGA&Season=2019-20&section=player&sct=hex",
            "x-nba-stats-token": "true",
            }
        proxies = {
            "http" : "http://203.202.245.62:80",
            "http" : "http://63.82.52.254:8080",
            } 
        json_data = requests.get(url, headers= kv, proxies= proxies, timeout= 10).json()
        return json_data
    except:
        return ""

def fillparselist(shot_data, col_headers, All_data):
    col_headers = All_data['resultSets'][0]['headers']
    shot_data = All_data['resultSets'][0]['rowSet']
    df = pd.DataFrame(shot_data, columns= col_headers)
    return(df)

def savefile(data_df):
    #data_df.to_csv('/Users/Marhoo/NBA-Streamlit/player_shot_datasets/shot_data.csv', index= False)
    print('保存成功')

def main():
    shotchart_data = []
    column_headers = []
    url = 'https://stats.nba.com/stats/shotchartdetail?AheadBehind=&CFID=33&CFPARAMS=2019-20&ClutchTime=&Conference=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&Division=&EndPeriod=10&EndRange=28800&GROUP_ID=&GameEventID=&GameID=&GameSegment=&GroupID=&GroupMode=&GroupQuantity=5&LastNGames=0&LeagueID=00&Location=&Month=0&OnOff=&OpponentTeamID=0&Outcome=&PORound=0&Period=0&PlayerID=201935&PlayerID1=&PlayerID2=&PlayerID3=&PlayerID4=&PlayerID5=&PlayerPosition=&PointDiff=&Position=&RangeType=0&RookieYear=&Season=2019-20&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StartPeriod=1&StartRange=0&StarterBench=&TeamID=0&VsConference=&VsDivision=&VsPlayerID1=&VsPlayerID2=&VsPlayerID3=&VsPlayerID4=&VsPlayerID5=&VsTeamID='
    All_data = getshotchart(url)
    df = fillparselist(shotchart_data, column_headers, All_data)
    savefile(df)
main()
