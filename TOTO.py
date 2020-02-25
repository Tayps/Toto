from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

site = 'https://en.lottolyzer.com/history/singapore/toto/page/1/per-page/556/summary-view'
response = urlopen(site).read()
soup = BeautifulSoup(response, 'html.parser')

name_box = soup.find('table', attrs={'class': 'responsive-table'})
# name = name_box.text.strip()
namerows = name_box.find_all('tr')

l = []
for tr in namerows:
    td = tr.find_all('td')
    row = [tr.text.strip() for tr in td]
    l.append(row)
df = pd.DataFrame(l, columns=['Draw', 'Date', 'Winning_No', 'Addl_No', 'From_Last', 'Sum', 'Average', 'Odd_Even', '1-10', '11-20', '21-30', '31-40', '41-50'])
df = df.iloc[2:]
# df[['A', 'B', 'C', 'D', 'E', 'F',]] = df.Winning_No.str.split(",",expand=True)
df = df.set_index('Draw').Winning_No.str.split(",",expand=True).stack()
df = pd.get_dummies(df, prefix = 'Number').groupby(level=0).sum()

freq_draws = apriori(df, min_support=0.018, use_colnames=True)
rules = association_rules(freq_draws, metric = "lift", min_threshold=1)
pd.set_option('display.max_columns', None)
rules.head()
rules.to_csv('toto_mba.csv')
