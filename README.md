# Purchase Intent Prediction

## 📌 Project Overview

This project aims to predict customer purchase intent based on consumer electronics sales data. It leverages machine learning pipelines for feature engineering, model training, and deployment. The model is deployed using **FastAPI** and **AWS Lambda**, enabling efficient inference.

## 🛠️ Technologies Used

- **Python** (pandas, numpy, scikit-learn)
- **Machine Learning Pipeline** (Feature Engineering, Preprocessing, Model Training)
- **FastAPI** (API Development)
- **AWS Lambda** (Serverless Deployment)
- **Joblib** (Model Serialization)
- **MLOps** (Automated ML workflows)

## 📂 Project Structure

```
MLOPS/
├── dependencies/
│   ├── aws_lambda_artifact.zip   # Deployment package for AWS Lambda
│   ├── consumer_electronics_sales_data.csv  # Dataset
├── fast.py                        # FastAPI application
├── le_brand.pkl                   # LabelEncoder for ProductBrand
├── le_category.pkl                 # LabelEncoder for ProductCategory
├── pipeline.py                     # Model training pipeline
├── purchase_intent_model.pkl       # Trained machine learning model
├── requirements.txt                # Python dependencies
```

## 🚀 How to Run Locally

### 1️⃣ Setup Environment

```sh
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```sh
python pipeline.py
```

This script:
- Reads the dataset
- Cleans and preprocesses data
- Builds an ML pipeline with feature selection
- Trains a **RandomForestClassifier**
- Saves the model and encoders

### 3️⃣ Start FastAPI Server

```sh
uvicorn fast:app --reload
```

FastAPI will be running at: `http://127.0.0.1:8000`

## 🔧 Deploy to AWS Lambda

### 1️⃣ Package Dependencies

```sh
pip install -r requirements.txt -t package/
cd package && zip -r ../aws_lambda_artifact.zip .
```

### 2️⃣ Add Model Files

```sh
zip -g aws_lambda_artifact.zip purchase_intent_model.pkl
zip -g aws_lambda_artifact.zip le_category.pkl le_brand.pkl
zip -g aws_lambda_artifact.zip fast.py
```

### 3️⃣ Upload to AWS Lambda

- Navigate to AWS Lambda console
- Create a new function (Python 3.x)
- Upload `aws_lambda_artifact.zip`
- Configure API Gateway for invocation

## 📊 Model Performance

The model achieved an accuracy of **XX%** on the test set.

## 🤝 Contributing

Feel free to fork this repo and submit a PR with improvements!

## 📜 License

MIT License

---

🚀 **Developed by SUHIB ALFURJANI**

