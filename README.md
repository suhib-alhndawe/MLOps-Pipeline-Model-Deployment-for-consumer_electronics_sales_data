# Purchase Intent Prediction Project with FastAPI & RandomForest 🚀

## Project Description
This project aims to build a web interface for predicting customers' purchase intent in an electronics store, using a machine learning model trained on sales data. The project uses FastAPI as the web framework and RandomForest as the classification algorithm.

[**Application Link**](https://mlops-pipeline-model-deployment-for.onrender.com) 🌐

## Project Components
- **Python Libraries**: `pandas`, `numpy`, `sklearn`, `joblib`, `fastapi`, `starlette`
- **Model Used**: RandomForestClassifier 🧠
- **Services**: Deployed on Render 🌐

## Workflow Steps

### 1. Data Reading and Cleaning 🧹
- Importing data from a CSV file.
- Encoding categorical variables using `LabelEncoder`.
- Removing unnecessary columns.

### 2. Feature Engineering 🔍
- Adding a `PricePerUnit` feature by calculating price per unit.
- Categorizing ages into age groups using `pd.cut`.

### 3. Data Splitting 📊
- Splitting the data into training and testing sets (75/25 split).

### 4. Model Building with Pipeline 🛠️
- `MinMaxScaler` for data normalization.
- `SelectKBest` to select the top 5 features.
- `RandomForestClassifier` as the classification model.

### 5. Saving the Model and LabelEncoders 💾
The model and the `LabelEncoders` are saved using the `joblib` library.

### 6. Frontend Development 🎨
- Creating a web page using HTML, CSS, and JavaScript.
- Collecting user inputs via a form and sending them to the API.

### 7. Running the Backend with FastAPI ⚙️
- `/PurchaseIntent` endpoint for prediction.
- Handling user inputs and converting them to model-compatible format.

## How to Use 📖

1. Run the project locally:
```bash
uvicorn main:app --reload
```

### API Request Example ⚡
```bash
GET /PurchaseIntent?PrCa=Smart%20Watches&PrBr=Samsung&PrPr=980.39&CuAge=35&CuGe=1&PuFr=7&CuSa=2
```

### Expected Response ✅
```json
{"message": "Purchase"} or {"message": "No Purchase"}
```

## Notes 📝
- Ensure the API is running before sending requests.
- Make sure text values match the ones the model was trained on.

**Developer**: Suhib Alfurjani 👨‍💻

