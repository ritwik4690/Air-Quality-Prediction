import pandas as pd
import datetime as dt
import pickle

from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.linear_model import LogisticRegression

stn_map = {'TG001':1, 'TG002':2, 'TG003':3, 'TG004':4, 'TG006':6, 'AP005':5}
def conv_input(date_in, sid):
    date_in = date_in.toordinal()
    sid = stn_map[sid]
    return [sid,date_in]


df = pd.read_excel('station_day.csv')
df = df.dropna()
df = df.drop(columns=['NO','NH3','SO2','O3','Benzene','Toluene','Xylene'])

sub_df = df[df.groupby('StationId').StationId.transform('count')>900].copy()
sub_df['Date'] = pd.to_datetime(sub_df['Date'], format="%Y-%M-%d")
sub_df['Date'] = sub_df['Date'].map(dt.datetime.toordinal)

station_map = { "StationId":{"TG001":1, "TG002":2, "TG003":3, "TG004":4, "AP005":5, 'TG006':6}}
sub_df=sub_df.replace(station_map)

x=sub_df.iloc[:,1:3]
y=sub_df.iloc[:,8:10]

regressor = LogisticRegression()

regressor.fit(x,y.iloc[:, 0:1])

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

input_date = '2040-01-08'
input_stn = 'AP005'
X_in = conv_input(pd.Timestamp(input_date),input_stn)

print(model.predict([X_in]))
