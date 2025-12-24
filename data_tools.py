import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path='data/superstore.csv'):
    df = pd.read_csv(file_path)
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    return df

def clean_data(df):
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if 'Sales' in df.columns:
        df = df[df['Sales'] < 10000]
    df = df.drop_duplicates()
    return df

def engineer_features(df):
    if 'Sales' in df.columns and 'Discount' in df.columns:
        df['Discounted Price'] = df['Sales'] * (1 - df['Discount'])
    if 'Order Date' in df.columns:
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
    return df

def transform_data(df, features=['Quantity', 'Discount', 'Month', 'Year']):
    X = df[features]
    y = df['Sales'] if 'Sales' in df.columns else None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')  
    return X_scaled, y
    import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path='data/superstore.csv'):
    df = pd.read_csv(file_path)
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    return df

def clean_data(df):
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if 'Sales' in df.columns:
        df = df[df['Sales'] < 10000]
    
    df = df.drop_duplicates()
    return df


def engineer_features(df):
   
    if 'Sales' in df.columns and 'Discount' in df.columns:
        df['Discounted Price'] = df['Sales'] * (1 - df['Discount'])
    
    if 'Order Date' in df.columns:
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
    return df


def transform_data(df, features=['Quantity', 'Discount', 'Month', 'Year']):
    
    X = df[features]
    y = df['Sales'] if 'Sales' in df.columns else None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')  
    return X_scaled, y
