from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
from mangum import Mangum
# تحميل النموذج المدرب و LabelEncoders المحفوظة
pipeline = joblib.load('purchase_intent_model.pkl')
le_category = joblib.load('le_category.pkl')
le_brand = joblib.load('le_brand.pkl')

# إعداد FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

@app.get("/", response_class=HTMLResponse)
def hi():
    return """
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PurchaseIntent</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(to right, #f9f9f9, #e0f7fa);
                margin: 0;
                padding: 0;
                animation: fadeIn 1s ease-in-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            header {
                background-color: #4CAF50;
                color: white;
                padding: 10px 0;
                text-align: center;
                font-size: 28px;
                animation: slideIn 0.5s forwards;
            }
            @keyframes slideIn {
                from { transform: translateY(-20px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .container {
                width: 80%;
                margin: 20px auto;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            input[type="text"], input[type="number"] {
                width: calc(100% - 20px);
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 5px;
                transition: border 0.3s;
            }
            input[type="text"]:focus, input[type="number"]:focus {
                border: 1px solid #4CAF50;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #45a049;
            }
            #response {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            footer {
                text-align: center;
                margin-top: 20px;
                font-size: 16px;
                color: #777;
                animation: fadeIn 1s ease-in-out 0.5s forwards;
            }
        </style>
    </head>
    <body>
        <header>
            Purchase Intent
        </header>
        <div class="container">
            <form id="purchaseIntentForm">
                <label for="productCategory">فئة المنتج:</label>
                <input type="text" id="productCategory" name="PrCa" value="Smart Watches" required>

                <label for="productBrand">علامة المنتج:</label>
                <input type="text" id="productBrand" name="PrBr" value="Samsung" required>

                <label for="productPrice">سعر المنتج:</label>
                <input type="number" id="productPrice" name="PrPr" value="980.39" required step="0.01">

                <label for="customerAge">عمر العميل:</label>
                <input type="number" id="customerAge" name="CuAge" value="35" required>

                <label for="customerGender">جنس العميل (1 للذكور، 0 للإناث):</label>
                <input type="number" id="customerGender" name="CuGe" value="1" required>

                <label for="purchaseFrequency">تكرار الشراء:</label>
                <input type="number" id="purchaseFrequency" name="PuFr" value="7" required>

                <label for="customerSatisfaction">رضا العميل:</label>
                <input type="number" id="customerSatisfaction" name="CuSa" value="2" required>

                <button type="submit">إرسال</button>
            </form>
            <div id="response"></div>
        </div>
        <footer>
            SUHIB ALFURJANI
        </footer>

        <script>
            document.getElementById('purchaseIntentForm').addEventListener('submit', async function(event) {
                event.preventDefault(); // منع إعادة تحميل الصفحة
                const formData = new FormData(this);
                const params = new URLSearchParams(formData).toString();
                const apiUrl = `http://127.0.0.1:8000/PurchaseIntent?${params}`;

                try {
                    const response = await fetch(apiUrl);
                    const data = await response.json();
                    document.getElementById('response').innerText = `الاستجابة: ${data.message}`;
                } catch (error) {
                    console.error('حدث خطأ:', error);
                    document.getElementById('response').innerText = 'حدث خطأ أثناء الاتصال بالـ API. تحقق من البيانات المدخلة وتأكد من أن الـ API قيد التشغيل.';
                }
            });
        </script>
    </body>
    </html>
    """
@app.get("/PurchaseIntent")
def predict_purchase(PrCa: str, PrBr: str, PrPr: float, CuAge: int, CuGe: int, PuFr: int, CuSa: int):
    try:
        input_data = [
            le_category.transform([PrCa])[0],    # ProductCategory
            le_brand.transform([PrBr])[0],       # ProductBrand
            PrPr,                                # ProductPrice
            CuAge,                               # CustomerAge
            CuGe,                                # CustomerGender
            PuFr,                                # PurchaseFrequency
            CuSa,                                # CustomerSatisfaction
            PrPr / PuFr,                         # PricePerUnit (Example of feature engineering)
            pd.cut([CuAge], bins=[0, 18, 35, 50, 100], labels=[0, 1, 2, 3])[0]  # Age_Category (Feature Engineering)
        ]
    except ValueError as e:
        return {"message": f"خطأ: {str(e)}"}

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    try:
        # Apply scaling and feature selection to the input data
        input_data_transformed = pipeline.named_steps['scaler'].transform(input_data_reshaped)
        input_data_selected = pipeline.named_steps['feature_selection'].transform(input_data_transformed)

        # Make prediction
        prediction = pipeline.named_steps['model'].predict(input_data_selected)
    except Exception as e:
        return {"message": f"Error during prediction: {str(e)}"}

    return {"message": "Purchase" if prediction[0] == 1 else "No Purchase"}
