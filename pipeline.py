import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import joblib

# =========================
# READ DATA SET
# =========================
df = pd.read_csv(r"consumer_electronics_sales_data.csv")

# =========================
# DATA CLEAN
# =========================
le_category = LabelEncoder()
le_brand = LabelEncoder()

df['ProductCategory'] = le_category.fit_transform(df['ProductCategory'])
df['ProductBrand'] = le_brand.fit_transform(df['ProductBrand'])
df.drop('ProductID', axis=1, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================
df['PricePerUnit'] = df['ProductPrice'] / df['PurchaseFrequency']  # Example of a new feature
df['Age_Category'] = pd.cut(df['CustomerAge'], bins=[0, 18, 35, 50, 100], labels=[0, 1, 2, 3])  # Age category

# =========================
# SPLIT DATA
# =========================
X = df.drop('PurchaseIntent', axis=1)
y = df['PurchaseIntent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# =========================
# PIPELINE WITH SCALING, FEATURE SELECTION & MODELING
# =========================
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Normalize the data to [0, 1] range
    ('feature_selection', SelectKBest(chi2, k=5)),  # Select top 5 features using chi-squared
    ('model', RandomForestClassifier(max_depth=10, random_state=0))  # Random Forest classifier
])

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Save the trained model and encoders
joblib.dump(pipeline, 'purchase_intent_model.pkl')
joblib.dump(le_category, 'le_category.pkl')  # Save the LabelEncoder for ProductCategory
joblib.dump(le_brand, 'le_brand.pkl')  # Save the LabelEncoder for ProductBrand

# Optionally: Evaluate the model on the test set
test_accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
