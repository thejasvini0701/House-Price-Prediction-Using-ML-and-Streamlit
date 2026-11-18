
# 🏡 House Price Prediction Using Machine Learning

## 📌 Project Overview

This project predicts house prices using Machine Learning.

It uses property details such as the number of bedrooms, bathrooms, living area, lot area, floors, and year built to estimate the price of a house.

The project includes an interactive web application built using Streamlit.

## 🎯 Project Objectives

- Understand and analyze housing data.
- Perform data cleaning and preprocessing.
- Train a Machine Learning model.
- Evaluate model performance.
- Predict house prices using user inputs.
- Build an interactive web application.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 🤖 Machine Learning Algorithm

**Linear Regression**

The model uses the following input features:

- Bedrooms
- Bathrooms
- Living Area (sqft)
- Lot Area (sqft)
- Floors
- Year Built

**Target Variable:** House Price

## 📊 Model Performance

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 164,083.68 |
| R² Score | 0.569 |

These are the results from the earlier test-set evaluation.

## ✨ Application Features

- Interactive dashboard
- Housing dataset overview
- House price distribution chart
- Average price by bedrooms
- Average price by floors
- Feature correlation heatmap
- House price prediction
- Model performance overview

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── app.py
├── train_model.py
├── data.csv
├── house_price_model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd House-Price-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

## 🚀 Future Improvements

- Experiment with more Machine Learning algorithms.
- Improve prediction accuracy.
- Add more housing features.
- Enhance dashboard visualizations.
- Deploy the application online.


## ⭐ Acknowledgement

Built using Python, Scikit-learn, and Streamlit.


## 🌐 Live Demo

Check out the live House Price Prediction application:

[🚀 Open House Price Predictor](https://house-price-prediction-using-ml-and-app-xsllepzz42oanx7dsxx4er.streamlit.app/)
