from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# --- 1. โหลดข้อมูลและแปลง One-Hot Encoding ---
clean_data = []
with open('Bkk_Real_Estate_data.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                prop_type = parts[0]
                beds = 1 if prop_type == 'Apartment' else int(parts[3])
                baths = 1 if prop_type == 'Apartment' else int(parts[4])
                clean_data.append({
                    'Type': prop_type, 'Location': parts[1], 'Area': float(parts[2]),
                    'Bedrooms': beds, 'Bathrooms': baths, 'Price': float(parts[5])
                })
            except ValueError:
                pass

df = pd.DataFrame(clean_data)
df_encoded = pd.get_dummies(df, columns=['Type', 'Location'])
features_cols = [col for col in df_encoded.columns if col != 'Price']

# --- 2. คำนวณ Linear Algebra ---
A = np.column_stack((np.ones(len(df_encoded)), df_encoded[features_cols].values * 1)).astype(float)
#log transformation
# แปลงราคา b ให้เป็น log(b) ก่อนนำไปหา weights
b_log = np.log(df['Price'].values.astype(float))
weights = np.linalg.pinv(A.T @ A) @ A.T @ b_log

# --- 3. วิเคราะห์ความคลาดเคลื่อน (แยกตามทำเล) ---
df['Predicted_Price'] = A @ weights
df['Abs_Error'] = np.abs(df['Price'] - df['Predicted_Price'])
df['Error_Percent'] = (df['Abs_Error'] / df['Price']) * 100

# จัดกลุ่มความคลาดเคลื่อนตามทำเลเก็บเป็น Dictionary
loc_errors = df.groupby('Location')[['Abs_Error', 'Error_Percent']].mean().to_dict('index')

# --- 4. สร้าง API ---
app = FastAPI(title="BKK Real Estate API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PropertyData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    prop_type: str
    location: str

@app.post("/predict")
def predict_price(data: PropertyData):
    if data.prop_type == 'Apartment':
        data.bedrooms = 1
        data.bathrooms = 1

    input_vector = np.zeros(len(features_cols))
    input_vector[features_cols.index('Area')] = data.area
    input_vector[features_cols.index('Bedrooms')] = data.bedrooms
    input_vector[features_cols.index('Bathrooms')] = data.bathrooms
    
    if f"Type_{data.prop_type}" in features_cols: 
        input_vector[features_cols.index(f"Type_{data.prop_type}")] = 1
    if f"Location_{data.location}" in features_cols: 
        input_vector[features_cols.index(f"Location_{data.location}")] = 1

    #log transformation
    # โมเดลจะทำนายออกมาเป็นค่า Log
    predicted_log = weights[0] + np.dot(weights[1:], input_vector)
    
    # แปลงกลับเป็นราคาบาทด้วย Exponential (np.exp)
    predicted_price = np.exp(predicted_log)
    
    # ดึงค่าความคลาดเคลื่อนของทำเลที่เลือก
    mae = loc_errors.get(data.location, {}).get('Abs_Error', 0)
    mape = loc_errors.get(data.location, {}).get('Error_Percent', 0)

    return {
        "status": "success", 
        "predicted_price_thb": round(predicted_price, 2),
        "location_mae": round(mae, 2),
        "location_mape": round(mape, 2)
    }