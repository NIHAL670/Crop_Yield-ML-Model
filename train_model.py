import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv("india_crop_yield_20000_rows.csv")

# Extra features
if "Nitrogen" not in data.columns:
    data["Nitrogen"] = np.random.randint(40,120,size=len(data))
    data["Phosphorus"] = np.random.randint(20,60,size=len(data))
    data["Potassium"] = np.random.randint(20,60,size=len(data))
    data["Humidity"] = np.random.randint(40,90,size=len(data))

soil_types = ["Alluvial","Black","Red","Laterite","Clay"]
data["Soil_Type"] = np.random.choice(soil_types,len(data))

# Create yield column
data["Yield"] = data["Production"] / data["Area"]

# Encode categorical data
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()
le_soil = LabelEncoder()

data["State"] = le_state.fit_transform(data["State"])
data["Crop"] = le_crop.fit_transform(data["Crop"])
data["Season"] = le_season.fit_transform(data["Season"])
data["Soil_Type"] = le_soil.fit_transform(data["Soil_Type"])

# Features
X = data[['State','Crop','Season','Soil_Type',
          'Area','Rainfall','Temperature',
          'Humidity','Nitrogen','Phosphorus','Potassium']]

y = data["Yield"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X,y)

# Save model and encoders
joblib.dump(model,"model.pkl")
joblib.dump(le_state,"le_state.pkl")
joblib.dump(le_crop,"le_crop.pkl")
joblib.dump(le_season,"le_season.pkl")
joblib.dump(le_soil,"le_soil.pkl")

print("Model saved successfully!")