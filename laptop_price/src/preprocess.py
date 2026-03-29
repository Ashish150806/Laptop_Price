"""
Preprocessing utilities for laptop price prediction.
These functions mirror the transformations applied in the notebook.
"""
import numpy as np


def fetch_processor(text):
    """Categorize CPU into brand groups."""
    if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return text
    elif text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'


def cat_os(inp):
    """Categorize OS into groups."""
    if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    elif inp in ('macOS', 'Mac OS X'):
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


def extract_ppi(screen_size, resolution):
    """Calculate pixels per inch from screen size and resolution string."""
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
    return ppi


def preprocess_dataframe(df):
    """Apply all preprocessing steps to raw laptop dataframe."""
    df = df.copy()
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

    df['X_res'] = df['ScreenResolution'].apply(lambda x: x.split()[-1].split('x')[0]).astype('int')
    df['Y_res'] = df['ScreenResolution'].apply(lambda x: x.split()[-1].split('x')[1]).astype('int')
    df['ppi'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5 / df['Inches']).astype('float')
    df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
    df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

    df.drop(columns=['Memory'], inplace=True)

    df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df[df['Gpu brand'] != 'ARM']
    df.drop(columns=['Gpu'], inplace=True)

    df['os'] = df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'], inplace=True)

    return df
