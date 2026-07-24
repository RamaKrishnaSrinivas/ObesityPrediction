# ObesityPrediction

# AI-Based Obesity Level Prediction System

## 📌 Project Overview

The AI-Based Obesity Level Prediction System is a Machine Learning project that predicts a person's obesity level based on their physical characteristics and lifestyle habits.

The system uses the Obesity Dataset and a Random Forest Classifier to predict different obesity categories. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and web deployment using Flask.

---

## 🎯 Objective

The main objectives of this project are:

- To analyze obesity-related data.
- To perform Exploratory Data Analysis (EDA).
- To clean and preprocess the dataset.
- To train a Machine Learning classification model.
- To evaluate the performance of the model.
- To predict obesity levels for new user inputs.
- To deploy the application as a web application using Flask.

---

## 📊 Dataset

The project uses:

**ObesityDataSet_raw_and_data_sinthetic.csv**

The dataset contains information related to:

- Gender
- Age
- Height
- Weight
- Family history of overweight
- High-calorie food consumption
- Vegetable consumption
- Number of meals
- Eating between meals
- Smoking
- Water consumption
- Calorie monitoring
- Physical activity
- Technology usage
- Alcohol consumption
- Transportation

### Target Variable

`NObeyesdad`

The target variable represents the obesity level of a person.

---

## 🧠 Machine Learning Model

The project uses:

### Random Forest Classifier

Random Forest is an ensemble Machine Learning algorithm that combines multiple Decision Trees to make a final prediction.

The model is trained using the input features and predicts the obesity category of a new person.

---

## 🔄 Machine Learning Workflow

```text
Dataset
    ↓
Data Loading
    ↓
Exploratory Data Analysis
    ↓
Data Cleaning
    ↓
Label Encoding
    ↓
Feature Scaling
    ↓
Train-Test Split
    ↓
Random Forest Classifier
    ↓
Model Evaluation
    ↓
New Input Prediction
    ↓
Flask Web Application
