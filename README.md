# Customer Churn Prediction using machine learning

## Project Overview
Customer churn is a major challenge for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones. This project aims to build a machine learning model that predicts whether a customer will churn based on demographic information, service usage, and billing details

The goal is to identify high-risk customers so that businesses can take proactive retention actions.

---

## Dataset

Dataset: **"Telco Customer Churn Dataset"**

The dataset contains information about **7,043 telecom customers** with **21 features** including demographics, service subscriptions, and billing information.

### Key features

|Feature      |Description    |
|-------------|-------------------------|
|tenure |Number of months the customer has stayed with the company |
| MonthlyCharges | Monthly service charges | 
| TotalCharges | Total amount charged to the customer | 
| Contract | Type of contract (Month-to-month, One year, Two year) | 
| InternetService | Internet service provider type | 
| PaymentMethod | Payment method used by the customer | 
| Churn | Target variable indicating whether the customer churned |

---

## Project Workflow

### 1.Data Cleaning

* Converted TotalCharges to numeric
* Handled missing values
* Removed inconsistencies in the dataset

### 2. Exploratory Data Analysis (EDA) 

EDA was performed to understand customer behavior and churn patterns. 

Key analyses included: 
* Churn distribution 
* Contract type vs churn 
* Tenure distribution 
* Monthly charges vs churn 
* Correlation analysis

### 3.Feature Engineering

* Removed unnecessary columns
* One-hot encoding for categorical variables
* Train-test split

### 4.Model Training

Handling class imbalance, Customer churn datasets are typically imbalanced.

Techniques used:
* Class weighting 
* SMOTE (Synthetic Minority Oversampling Technique)

Several models were trained and compared

* Logistic Regression(Class Balanced)
* Logistic Regression with SMOTE
* Random Forest
* Random Forest tuned
* XGBoost


### Model Evaluation

Models were evaluated using multiple metrics

* Accuracy
* Recall
* F1 score
* Precision
* ROC-AUC

Model performance

| Model  |Accuracy  |Recall  |F1 score  |Precision   | ROC-AUC  |
|--------|----------|--------|----------|------------|----------|
|Logistic Regression(Balanced)|~73%	| ~80%	| ~61%	| ~49%| 0.83 |
|Logistic Regression with SMOTE| ~73% | ~78%	| ~61%	|~50%| 0.83 |
| Random Forest| ~79%	| ~50%	| ~56%	| ~62% | 0.82 |
| Random Forest tuned| ~76%	| ~75%	| ~62%	| ~53%| 0.84 |
| XGBoost| ~76%	| ~54%	| ~54%	| ~55% |  0.80 |
| Final model| **~76%**	| **~75%**	| **~62%**	| **~53%** |  **0.84** |

The ROC-AUC score of **0.84** indicates strong capability in distinguishing churned customers from non-churned customers.

## Feature importance 

The most influential factors affecting customer churn were:


