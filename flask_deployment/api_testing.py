import pandas as pd
import random
import requests

df = pd.DataFrame(index = range(100), columns=['vol_moving_avg', 'adj_close_rolling_med'])

df['vol_moving_avg'] = df['vol_moving_avg'].apply(lambda x: random.randint(0, 2326635323))
df['adj_close_rolling_med'] = df['adj_close_rolling_med'].apply(lambda x: random.randint(-100000, 100000))

for index, row in df.iterrows():
    endpoint = f"https://stock-volume-predict.onrender.com/predict?vol_moving_avg={row['vol_moving_avg']}&adj_close_rolling_med={row['adj_close_rolling_med']}"
    response = requests.get(endpoint)
    df.loc[index, 'vol_prediction'] = response.text
    df.loc[index, 'duration (in seconds)'] = response.elapsed.total_seconds()

df.to_csv('API_testing.csv', index=False)