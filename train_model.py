import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the data"""
    df = pd.read_csv('Mumbai1.csv')
    
    # Drop irrelevant columns
    df = df.drop(['Unnamed: 0', 'New/Resale', 'Intercom', 'Gas Connection', 'Jogging Track', 
                  'Swimming Pool', 'Clubhouse', 'School', 'Gymnasium'], axis=1, errors='ignore')
    
    # Convert 'Area' to numerical
    df['Area'] = df['Area'].astype(str).str.replace(' sq.ft.', '').astype(float)
    
    # Convert 'Price' to numerical
    def convert_price(price_str):
        price_str = str(price_str)
        if ' Lacs' in price_str:
            return float(price_str.replace(' Lacs', '').replace(',', '')) * 100000
        elif ' Cr' in price_str:
            return float(price_str.replace(' Cr', '').replace(',', '')) * 10000000
        return float(price_str.replace(',', ''))
    
    df['Price'] = df['Price'].apply(convert_price)
    
    # Feature engineering
    df['Price_per_sqft'] = df['Price'] / df['Area']
    df['Location'] = df['Location'].astype(str).str.strip().str.title()
    
    # Handle outliers in price using IQR
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]
    
    # Location grouping
    location_counts = df['Location'].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df['Location'] = df['Location'].replace(rare_locations, 'Other')
    
    # Add location clusters based on price per sqft
    location_stats = df.groupby('Location')['Price_per_sqft'].agg(['mean', 'count'])
    location_stats['Location_Cluster'] = pd.qcut(location_stats['mean'], q=5, labels=False, duplicates='drop')
    location_map = location_stats['Location_Cluster'].to_dict()
    df['Location_Cluster'] = df['Location'].map(location_map)
    
    return df

def create_model(df):
    """Create and train the model"""
    # Select features and target
    X = df[['Location', 'Area', 'No. of Bedrooms', 'Car Parking', 'Price_per_sqft', 'Location_Cluster']]
    y = df['Price']
    
    # Preprocessing pipeline
    categorical_features = ['Location', 'Car Parking']
    numerical_features = ['Area', 'No. of Bedrooms', 'Price_per_sqft', 'Location_Cluster']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=5))
            ]), categorical_features)
        ])
    
    # Use Random Forest
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train model
    model.fit(X, y)
    return model

def get_form_options(df):
    """Get options for the web form"""
    locations = sorted(df['Location'].unique())
    car_parking = sorted(df['Car Parking'].unique())
    bedrooms = sorted(df['No. of Bedrooms'].unique())
    
    # Calculate typical price per sqft based on location clusters
    cluster_stats = df.groupby('Location_Cluster')['Price_per_sqft'].mean().round(2).to_dict()
    
    return {
        'locations': locations,
        'car_parking': car_parking,
        'bedrooms': bedrooms,
        'cluster_stats': cluster_stats
    }

if __name__ == '__main__':
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Training model...")
    model = create_model(df)
    
    print("Generating form options...")
    options = get_form_options(df)
    
    print("Saving model and options...")
    # Save the model
    joblib.dump(model, 'house_price_model.pkl')
    
    # Save options as JSON for the web app
    import json
    with open('form_options.json', 'w') as f:
        json.dump(options, f, indent=2)
    
    print("Model and options saved successfully!")
    print("Files created:")
    print("- house_price_model.pkl")
    print("- form_options.json")