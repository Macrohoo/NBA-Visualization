import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import re

def geturlhtml(url):
    try:
        kv = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1 Safari/605.1.15"}
        proxies = {
            "http" : "http://128.199.67.15:8080",
            "http" : "http://92.222.180.156:8080",
        }
        r = requests.get(url, headers= kv, proxies= proxies, timeout= 10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""

def fillparselist(col_headers, pla_data, html):
    soup = BeautifulSoup(html, 'html.parser')
    
    col_headers= []
    tr = soup.findAll('tr', limit= 2)[1]
    for th in tr.findAll('th'):
        col_headers.append(th.getText())
    col_headers.remove('Rk')

    data_rows = soup.findAll('tr')[2:]
    pla_data = []
    for i in range(len(data_rows)):
        player_row = []
        for td in data_rows[i].findAll('td'):
            player_row.append(td.getText())
        pla_data.append(player_row)
    
    df = pd.DataFrame(pla_data, columns= col_headers)
    df = df[df.Player.notnull()] #去掉数据中的nonetype行
    df.rename(columns ={'WS/48' :'WS_per_48'}, inplace=True)
    df.columns = df.columns.str.replace('%', '_Perc')
    df.columns.values[14:18] = [df.columns.values[14:18][col]+ "_per_G" for col in range(4)] #修改列名3连发
    #df = df[:].fillna(0) # index all the columns and fill in the 0
    #df = df.convert_objects(convert_numeric = True) #将具有数字字符串的列转换为最合适的数字数据类型
    #df.loc[:, 'Yrs':'AST'] = df.loc[:, 'Yrs':'AST'].astype(int)
    #df.append(0, 'Draft_Yr', span)
    return(df)
  
def savefile(data_df):
    data_df.to_csv('/Users/Marhoo/Desktop/testnba/All_drafts.csv', index = False)
    print('保存成功')

def main():
    go = 2016
    end = 2020
    All = pd.DataFrame()
    column_headers = []
    player_data = []
    url_start = 'https://www.basketball-reference.com/draft/NBA_{year}.html'
    count = 0
    for i in range(go, end):
        try:
            url = url_start.format(year = i)
            html = geturlhtml(url)
            df = fillparselist(column_headers, player_data, html)
            All = All.append(df, ignore_index = True)
            count = count + 1
            print('\r当前进度: {:.2f}%'.format(count*100/(end-go)), end= '')
        except:
            count = count + 1
            print('\r当前进度: {:.2f}%'.format(count*100/(end-go)), end= '')
            continue
    savefile(All)
main()
