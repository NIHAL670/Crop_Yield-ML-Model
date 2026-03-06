import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset

data = pd.read_csv("india_crop_yield_20000_rows.csv")

# Add extra features if missing

if "Nitrogen" not in data.columns:
 np.random.seed(42)
data["Nitrogen"] = np.random.randint(40,120,size=len(data))
data["Phosphorus"] = np.random.randint(20,60,size=len(data))
data["Potassium"] = np.random.randint(20,60,size=len(data))
data["Humidity"] = np.random.randint(40,90,size=len(data))

# Soil types

soil_types = ["Alluvial","Black","Red","Laterite","Clay"]

if "Soil_Type" not in data.columns:
 np.random.seed(42)
data["Soil_Type"] = np.random.choice(soil_types,len(data))

# Create yield column

data["Yield"] = data["Production"] / data["Area"]

# Label Encoding

le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()
le_soil = LabelEncoder()

data["State"] = le_state.fit_transform(data["State"])
data["Crop"] = le_crop.fit_transform(data["Crop"])
data["Season"] = le_season.fit_transform(data["Season"])
data["Soil_Type"] = le_soil.fit_transform(data["Soil_Type"])

# Feature columns

X = data[['State','Crop','Season','Soil_Type',
'Area','Rainfall','Temperature',
'Humidity','Nitrogen','Phosphorus','Potassium']]

y = data["Yield"]

# Train model

model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

model.fit(X,y)

# Save model and encoders

joblib.dump(model,"model.pkl",compress=3)
joblib.dump(le_state,"le_state.pkl",compress=3)
joblib.dump(le_crop,"le_crop.pkl")
joblib.dump(le_season,"le_season.pkl")
joblib.dump(le_soil,"le_soil.pkl")

print("✅ Model and encoders saved successfully!")
