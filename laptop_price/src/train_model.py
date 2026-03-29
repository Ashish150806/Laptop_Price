"""
Run this script once to generate df.pkl and pipe.pkl from laptop_data.csv.
Usage: python src/train_model.py  (run from project root)
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
df = pd.read_csv('data/laptop_data.csv')  # run from project root
df.drop(columns=['Unnamed: 0'], inplace=True)

# Clean Ram and Weight
df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

# Touchscreen
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

# IPS
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Resolution -> ppi
df['X_res'] = df['ScreenResolution'].apply(lambda x: x.split()[-1].split('x')[0]).astype('int')
df['Y_res'] = df['ScreenResolution'].apply(lambda x: x.split()[-1].split('x')[1]).astype('int')
df['ppi'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5 / df['Inches']).astype('float')
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Cpu brand
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))


def fetch_processor(text):
    if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# Drop Memory
df.drop(columns=['Memory'], inplace=True)

# Gpu brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)


# OS category
def cat_os(inp):
    if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    elif inp in ('macOS', 'Mac OS X'):
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

# Features and target
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Pipeline: OneHotEncode categorical columns [Company, TypeName, Cpu brand, Gpu brand, os]
# These are columns at indices 0, 1, 7, 8, 9 in X
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 8, 9])
], remainder='passthrough')

step2 = RandomForestRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print(f'R2 score: {r2_score(y_test, y_pred):.4f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')

# Export
pickle.dump(df, open('models/df.pkl', 'wb'))
pickle.dump(pipe, open('models/pipe.pkl', 'wb'))
print('Saved models/df.pkl and models/pipe.pkl')
