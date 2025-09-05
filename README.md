# Stock Market Prediction (Machine Learning Project)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)  
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-orange.svg)  
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## Project Overview
This repository contains a **Machine Learning project** that predicts stock price movements for **Tesla (TSLA)** based on historical stock data.  
The project applies **feature engineering, data visualization, and classification models** to forecast whether the next day's closing price will go up or down.

The dataset used is Tesla stock data (`TSLA.csv`) from Yahoo Finance.

## Key Features
- **Data Preprocessing & Cleaning:** Handles missing values, removes redundant columns, and extracts date features (day, month, year, quarter).
- **Exploratory Data Analysis (EDA):** Includes distribution plots, box plots, correlation heatmaps, and time-series visualizations of stock prices.
- **Feature Engineering:** Creates derived features such as `open-close`, `low-high`, and `is_quarter_end`.
- **Target Variable Creation:** Defines target as 1 if the next day's closing price is higher, otherwise 0.
- **Model Training & Evaluation:** Trains multiple models including Logistic Regression, Support Vector Classifier (SVC), and XGBoost, and compares their performance using ROC-AUC and confusion matrices.

## Repository Contents
- `stock_market_prediction.py` – Main Python script implementing the project.
- `TSLA.csv` – Tesla stock dataset (not included, fetch from Yahoo Finance).
- `README.md` – Project documentation.

## Data Visualization
- Tesla closing price trends
- Feature distributions (Open, High, Low, Close, Volume)
- Box plots to detect outliers
- Quarterly average stock trends
- Heatmap showing correlation between features
- Pie chart showing target variable distribution

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Download Tesla stock data (`TSLA.csv`) from Yahoo Finance and place it in the working directory.

## Usage

Run the script:
```bash
python stock_market_prediction.py
```

Modify dataset path or model parameters as needed.

## Example Output

### Model Evaluation Results
```
LogisticRegression() :
Training Accuracy :  0.983
Validation Accuracy :  0.744

SVC(kernel='poly', probability=True) :
Training Accuracy :  0.996
Validation Accuracy :  0.731

XGBClassifier() :
Training Accuracy :  0.999
Validation Accuracy :  0.756
```

The script also generates visualizations including:
- Confusion matrices for each model
- ROC curves comparing model performance
- Various stock data visualizations

## Dataset Format
The expected dataset format (`TSLA.csv`):
```csv
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,86.52,90.80,84.34,88.60,88.60,47660500
2020-01-03,88.99,90.31,87.00,88.41,88.41,34309500
...
```

## Tools and Technologies
- Python 3.x
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Data visualization
- scikit-learn - Machine learning models and evaluation
- XGBoost - Advanced boosting-based classification

## Author
Manoj Deepan M


