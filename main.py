from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')


d1 = {'A': [1, 2, 3],
      'B': [10, 12, 15],
      'ID': [1, 3, 2],
      'time': ['2018-03-13 00:00:00', '2018-02-12 00:00:00', '2018-07-17']
      }
d2 = {'M': [1, 2, 3],
      'N': [10, 12, 15],
      'ID': [3, 2, 1],
      'time': ['2018-02-12', '2018-07-17 00:00:00', '2018-03-13']
      }
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df1['time'] = pd.to_datetime(df1['time'])
df2['time'] = pd.to_datetime(df2['time'])

#df = pd.merge(df1, df2,  how='outer', on=['ID','time'])


df = pd.merge(df1, df2, on=['ID', 'time'])

print(df)
print(df.isna().sum())

'''
active_boards_df = pd.read_csv('./data/active_boards.csv')
#active_boards_df['time'] = active_boards_df['time'].map(lambda time: datetime.strptime(time, "%Y-%m-%d"))
active_boards_df['time'] = pd.to_datetime(active_boards_df['time'])
print(active_boards_df.head(10))
'''
cols = ['A', 'B']
df_copy = df[cols]
print(df_copy.head())
print(df_copy.shape)
print(df_copy.info())
col = pd.DataFrame(df_copy.columns)

print(col)


def test(x):
    return x


thresh_string2 = '2019-03-31'
thresh_string1 = '2019-03-29'
# Create date object in given time format yyyy-mm-dd
thresh_date1 = datetime.strptime(thresh_string1, "%Y-%m-%d")
thresh_date2 = datetime.strptime(thresh_string2, "%Y-%m-%d")
print((thresh_date2-thresh_date1).days)


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
cv = KFold(n_splits=3, random_state=0)

for train_index, test_index in cv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)



print(X.dtype)
