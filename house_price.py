import numpy as np
import pandas as pd

print("--- 1. โหลดและเตรียมข้อมูล ---")
clean_data = []
with open('Bkk_Real_Estate_data.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 0: continue # ข้ามบรรทัดหัวตาราง
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                # ข้อมูลใหม่เรียงตาม: Type, Location, Area, Bedrooms, Bathrooms, Price
                prop_type = parts[0]
                location = parts[1]
                area = float(parts[2])
                beds = int(parts[3])
                baths = int(parts[4])
                price = float(parts[5])
                
                clean_data.append([prop_type, location, area, beds, baths, price])
            except ValueError:
                pass 

# สร้าง DataFrame เบื้องต้น
df = pd.DataFrame(clean_data, columns=['Type', 'Location', 'Area', 'Bedrooms', 'Bathrooms', 'Price'])

# ==========================================
print("\n--- 2. สถิติข้อมูลเบื้องต้น ---")
print(f"จำนวนข้อมูลทั้งหมด: {len(df)} ชุด")
print(f"ประเภทอสังหาฯ ที่มี: {', '.join(df['Type'].unique())}")
print(f"ทำเลที่ตั้งที่มี: {', '.join(df['Location'].unique())}")
print(f"▶ ราคาเฉลี่ย:   {df['Price'].mean():,.2f} บาท  (ต่ำสุด: {df['Price'].min():,.0f} / สูงสุด: {df['Price'].max():,.0f})")
print(f"▶ พื้นที่เฉลี่ย:      {df['Area'].mean():.2f} ตร.ม.   (ต่ำสุด: {df['Area'].min():.0f} / สูงสุด: {df['Area'].max():.0f})")

# ==========================================
print("\n--- 3. ประมวลผล Linear Algebra ---")
# แปลงตัวหนังสือ (Type, Location) ให้เป็นเวกเตอร์เลข 0 และ 1 (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['Type', 'Location'], drop_first=False)

# ดึงชื่อคอลัมน์ที่เป็นตัวแปรต้น (ตัด Price ออก)
features = [col for col in df_encoded.columns if col != 'Price']

# เตรียมเมทริกซ์ A และเวกเตอร์ b
# เติม .astype(float) เพื่อบังคับให้ข้อมูลทุกตัวเป็นตัวเลขทศนิยม
A = np.column_stack((
    np.ones(len(df_encoded)),       
    df_encoded[features].values * 1 
)).astype(float)

b = df_encoded['Price'].values.astype(float)
# คำนวณหาค่าน้ำหนัก (Weights) ด้วย Pseudo-Inverse
weights = np.linalg.pinv(A.T @ A) @ A.T @ b

# ==========================================
print("\n=== กรุณาใส่ข้อมูลเพื่อทำนายราคา ===")
prop_type = input(f"1. ใส่ประเภทอสังหาฯ (เช่น Condo, House, Apartment) : ")
location = input(f"2. ใส่ทำเล (เช่น Sukhumvit, Silom, Thonglor) : ")
area = float(input("3. ใส่ขนาดพื้นที่ (ตร.ม.) : "))

# กฎสำหรับรับ Input: เช็คว่าเป็น Apartment หรือไม่
if prop_type == 'Apartment':
    print("*(ระบบตั้งค่าให้ Apartment มี 1 ห้องนอน 1 ห้องน้ำ อัตโนมัติ)*")
    bed = 1
    bath = 1
else:
    bed = float(input("4. ใส่จำนวนห้องนอน : "))
    bath = float(input("5. ใส่จำนวนห้องน้ำ : "))

# ==========================================
# สร้างเวกเตอร์สำหรับค่าที่ผู้ใช้กรอกเข้ามา เพื่อนำไปคูณกับเมทริกซ์น้ำหนัก
input_vector = np.zeros(len(features))
input_vector[features.index('Area')] = area
input_vector[features.index('Bedrooms')] = bed
input_vector[features.index('Bathrooms')] = bath

# เปิดสวิตช์ (ใส่ค่า 1) ในเวกเตอร์สำหรับประเภทและทำเลที่ผู้ใช้พิมพ์
if f"Type_{prop_type}" in features:
    input_vector[features.index(f"Type_{prop_type}")] = 1
if f"Location_{location}" in features:
    input_vector[features.index(f"Location_{location}")] = 1

# คำนวณราคาทำนาย: w0 + (w_other * input_vector)
predicted_price = weights[0] + np.dot(weights[1:], input_vector)

print("\n" + "=" * 50)
print(f"🎯 ราคาบ้านที่ทำนายได้ : {predicted_price:,.2f} บาท")
print("=" * 50)

# ==========================================
# การประเมินความคลาดเคลื่อน (Vector Norm)
b_hat = A @ weights
error_vector = b - b_hat
error_norm = np.linalg.norm(error_vector)
rmse = error_norm / np.sqrt(len(df))

print("\n--- 4. การประเมินความคลาดเคลื่อน (Vector Norm) ---")
print(f"ค่า L2-Norm ของความคลาดเคลื่อน: {error_norm:,.2f}")
print(f"สรุป: โมเดลทายราคาคลาดเคลื่อนเฉลี่ยหลังละประมาณ: {rmse:,.2f} บาท")

df['Predicted_Price'] = b_hat

# 2. คำนวณความคลาดเคลื่อนสัมบูรณ์ (Absolute Error) เป็นบาท
df['Abs_Error'] = np.abs(df['Price'] - df['Predicted_Price'])

# 3. คำนวณความคลาดเคลื่อนเป็นเปอร์เซ็นต์ (Percentage Error)
df['Error_Percent'] = (df['Abs_Error'] / df['Price']) * 100

print("\n--- 5. การวิเคราะห์ความคลาดเคลื่อนเชิงลึก (EDA) ---")
print("▶ ความคลาดเคลื่อนเฉลี่ย (MAE): {:,.2f} บาท".format(df['Abs_Error'].mean()))
print("▶ ความคลาดเคลื่อนเฉลี่ย (MAPE): {:.2f}%".format(df['Error_Percent'].mean()))

# 4. วิเคราะห์เจาะลึกแยกตาม "ประเภทอสังหาฯ"
print("\n--- เจาะลึกความคลาดเคลื่อนแยกตาม 'ประเภทอสังหาฯ' ---")
type_error = df.groupby('Type')[['Abs_Error', 'Error_Percent']].mean().round(2)
# เปลี่ยนชื่อคอลัมน์ให้แสดงผลสวยงาม
type_error.columns = ['พลาดไปเฉลี่ย (บาท)', 'คลาดเคลื่อน (%)']
print(type_error)

# 5. วิเคราะห์เจาะลึกแยกตาม "ทำเลที่ตั้ง"
print("\n--- เจาะลึกความคลาดเคลื่อนแยกตาม 'ทำเลที่ตั้ง' ---")
loc_error = df.groupby('Location')[['Abs_Error', 'Error_Percent']].mean().round(2)
loc_error.columns = ['พลาดไปเฉลี่ย (บาท)', 'คลาดเคลื่อน (%)']
# เรียงลำดับจากทำเลที่ทำนายพลาดมากที่สุดไปน้อยที่สุด
print(loc_error.sort_values('คลาดเคลื่อน (%)', ascending=False))