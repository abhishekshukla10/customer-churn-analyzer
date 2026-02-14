🔗 Live App: https://customer-churn-analyzer-29s6.onrender.com/

Built and deployed a production-ready churn prediction system using Logistic Regression (AUC 0.83) 
with business-optimized thresholding and risk segmentation, containerized via Docker and deployed on Render.

## Architecture
User Input
    ↓
Feature Encoding
    ↓
Scaler (StandardScaler)
    ↓
Logistic Regression
    ↓
Probability Output
    ↓
Threshold (0.3)
    ↓
Risk Segmentation
    ↓
Business Rule Layer
    ↓
Action Recommendation


🧠 Business Problem

Telecom companies face significant revenue loss due to customer churn.

The objective of this project is to:

  Predict customer churn probability
  Optimize decision thresholds based on business cost
  Segment customers into risk bands
  Recommend retention actions
  Deploy a production-ready ML system

This solution balances statistical performance with business feasibility.

##**Dataset
**
Telco Customer Churn Dataset

  ~7,000 customers
  21 original features
  Target variable: Churn (Yes/No)
  Imbalanced dataset (~26% churn rate)
**
🔎 Project Approach**
1️⃣ Exploratory Data Analysis

Identified strong predictors (Contract type, Tenure, Internet Service)

Detected multicollinearity (Tenure vs TotalCharges)

Evaluated churn distribution across service categories

2️⃣ Feature Engineering

One-hot encoding for categorical features

Removed high-cardinality ID column

Standardization using StandardScaler

3️⃣ Model Selection

Models evaluated:

Logistic Regression

Random Forest

Logistic Regression outperformed Random Forest in:

AUC

Recall optimization

Interpretability
**
📈 Model Performance**
Metric	Value
AUC	0.83
Recall (Churn)	~0.87 (threshold = 0.3)
Class Weight	Balanced

The final model was chosen for:

Strong ranking performance

Business-aligned recall

Interpretability

Deployment simplicity

**⚖️ Business Optimization Strategy
**
Instead of using default threshold (0.5):

Threshold tuned to 0.3

Balanced operational load and churn capture

Integrated business filtering logic

Risk Segmentation
Probability	Risk Level
≥ 0.6	High
0.3–0.6	Medium
0.1–0.3	Low
< 0.1	Stable

**##Business Rule Layer
**
Customers on 1-year or 2-year contracts are excluded from aggressive retention

High-risk customers → Immediate retention call

Medium-risk → Promotional outreach

Low-risk → Monitoring

This ensures cost-aware and customer-friendly intervention.
**
🐳 Deployment Architecture**

Model trained in Python (Scikit-learn)

Pipeline saved using Joblib

Streamlit UI built for interactive prediction

Dockerized application

Deployed on Render (Docker Web Service)

**
Tech Stack
**
Python
Pandas
Scikit-learn
Streamlit
Docker
Render (Cloud Deployment)
GitHub
**
📌 Key Learnings**

Business threshold tuning is critical in imbalanced problems

Model simplicity can outperform complex alternatives

Interpretability matters in churn prediction

Decision-layer logic is separate from model layer

Production ML requires structure + reproducibility
