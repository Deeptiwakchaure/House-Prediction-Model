#  Mumbai House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.pug.py)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



<p align="center">
  A web application that predicts house prices in Mumbai using machine learning. Users can input property details and get instant price predictions.
</p>

##  Features

- ** Accurate Predictions**: Uses Random Forest algorithm for house price estimation
- ** Location-Based**: Takes into account Mumbai's diverse neighborhoods and local pricing trends
- ** Data Visualization**: Comprehensive visualizations for data analysis
- ** Web Interface**: User-friendly form for easy input and prediction
- ** Input Validation**: Ensures data quality and prevents errors
- ** Responsive Design**: Works on both desktop and mobile devices

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, Random Forest |
| **Data Processing** | pandas, numpy |
| **Frontend** | HTML, CSS, Bootstrap 5 |
| **Model Storage** | joblib |
| **Serialization** | JSON |

## 📁 Project Structure
House-Prediction-Model/
├── 📄 train_model.py # Model training script
├── 📄 app.py # Flask web application
├── 📁 templates/
│ └── 📄 index.html # Web interface
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # This file
├── 📄 .gitignore # Git ignore rules
└── 📄 Mumbai1.csv # Dataset (not included in repo)


## 🎯 How It Works

1. **Data Preprocessing**: Cleans and prepares the housing data
2. **Feature Engineering**: Creates meaningful features like price per sq.ft.
3. **Model Training**: Uses Random Forest to learn from historical data
4. **Web Interface**: Provides a form for users to input property details
5. **Prediction**: Returns estimated house price based on input features

## 📊 Key Visualizations

The project includes several insightful visualizations:

- **Price Distribution**: Shows how house prices are distributed
- **Correlation Heatmap**: Displays relationships between numerical features
- **Location Price Boxplot**: Compares price distributions across locations
- **Actual vs Predicted**: Scatter plot showing model accuracy
- **Feature Importance**: Bar chart of most influential features
- **Residual Plot**: Helps identify prediction errors

##  Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- Mumbai1.csv dataset

### Installation

1. Clone the repository:
```bash
git clone git@github.com:Deeptiwakchaure/House-Prediction-Model.git
