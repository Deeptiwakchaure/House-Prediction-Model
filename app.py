from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load and preprocess the data
def load_and_preprocess_data():
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

# Create the model
def create_model(df):
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

# Load data and create model
print("Loading data and creating model...")
df = load_and_preprocess_data()
model = create_model(df)
print("Model created successfully!")

# Get unique values for form options
def get_form_options(df):
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

form_options = get_form_options(df)

@app.route('/')
def home():
    return render_template('index.html', options=form_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Location': [data['location']],
            'Area': [float(data['area'])],
            'No. of Bedrooms': [int(data['bedrooms'])],
            'Car Parking': [data['car_parking']],
            'Price_per_sqft': [float(data['price_per_sqft'])],
            'Location_Cluster': [int(data['location_cluster'])]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_price = prediction[0]
        
        return jsonify({
            'success': True,
            'predicted_price': f"₹{predicted_price:,.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)