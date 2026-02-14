print ("hi")

## Imports

import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import sklearn
print(sklearn.__version__)

## Data Loading

print("data loading start")
df=pd.read_csv(r"C:\Users\PGPBRDLT038\Desktop\PRJ\Customer Churn Analyzer\data\raw\telco_churn.csv")
print(f"shape of df is:{df.shape}")
print("data loading complete")
    

# Data cleaning
# convert total charges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("dropping null")
df = df.dropna()
print(f"shape of df is:{df.shape}")
# Drop customer ID, 
print("dropping customer ID")
df = df.drop([ 'customerID'], axis=1)
print(f"shape of df is:{df.shape}")

# Churn to numeric
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

## Model training definition

X = df.drop("Churn", axis=1)
print(f"shape of X is:{X.shape}")
y = df["Churn"]

X=pd.get_dummies(X,drop_first=True)
print(f"shape of X is:{X.shape}")

X_train, X_test, y_train,y_test=train_test_split(X,y, test_size=.2, random_state=42,stratify=y)

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=5  # replace if your best C is different
        ))
    ])

pipeline.fit(X_train, y_train)
feature_columns = X_train.columns.tolist()


print("model save start")

joblib.dump(pipeline, r"C:\Users\PGPBRDLT038\Desktop\PRJ\Customer Churn Analyzer\models\churn_model_pipeline.pkl")
joblib.dump(feature_columns, r"C:\Users\PGPBRDLT038\Desktop\PRJ\Customer Churn Analyzer\models\model_features.pkl")
print("model saved")

print("Model training complete and saved.")