import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
for estimator in model.estimators_:
    if not hasattr(estimator, "monotonic_cst"):
        estimator.monotonic_cst = None
le_state = joblib.load("le_state.pkl")
le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")
le_soil = joblib.load("le_soil.pkl")
# Glassmorphism UI styling (no logic changes)//


# Title
st.title("🌾 India Crop Yield Prediction System")

# Load dataset
@st.cache_data
def load_data():
    
    data = pd.read_csv("india_crop_yield_20000_rows.csv")
    return data
data = load_data()

# Add extra features (if not present)
if "Nitrogen" not in data.columns:
    data["Nitrogen"] = np.random.randint(40,120,size=len(data))
    data["Phosphorus"] = np.random.randint(20,60,size=len(data))
    data["Potassium"] = np.random.randint(20,60,size=len(data))
    data["Humidity"] = np.random.randint(40,90,size=len(data))

soil_types = ["Alluvial","Black","Red","Laterite","Clay"]
data["Soil_Type"] = np.random.choice(soil_types,len(data))

# Create yield column
data["Yield"] = data["Production"] / data["Area"]

# # Encode categorical data
# le_state = LabelEncoder()
# le_crop = LabelEncoder()
# le_season = LabelEncoder()
# le_soil = LabelEncoder()

# data["State"] = le_state.fit_transform(data["State"])
# data["Crop"] = le_crop.fit_transform(data["Crop"])
# data["Season"] = le_season.fit_transform(data["Season"])
# data["Soil_Type"] = le_soil.fit_transform(data["Soil_Type"])

# Features
X = data[['State','Crop','Season','Soil_Type',
          'Area','Rainfall','Temperature',
          'Humidity','Nitrogen','Phosphorus','Potassium']]

y = data["Yield"]

# Train model
# @st.cache_resource
# def train_model(X,y):
    
#     model = RandomForestRegressor(n_estimators=200)
#     model.fit(X,y)
#     return model
# model = train_model(X,y)

# UI Inputs
st.header("Enter Crop Details")

state = st.selectbox("State", le_state.classes_)
crop = st.selectbox("Crop", le_crop.classes_)
season = st.selectbox("Season", le_season.classes_)
soil = st.selectbox("Soil Type", le_soil.classes_)

area = st.number_input("Area (hectare)", min_value=1.0)

rainfall = st.slider("Rainfall (mm)", 0, 500, 200)
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 60)

nitrogen = st.slider("Nitrogen (N)", 0, 150, 90)
phosphorus = st.slider("Phosphorus (P)", 0, 100, 40)
potassium = st.slider("Potassium (K)", 0, 100, 40)

# Encode user input
state_enc = le_state.transform([state])[0]
crop_enc = le_crop.transform([crop])[0]
season_enc = le_season.transform([season])[0]
soil_enc = le_soil.transform([soil])[0]

# Prediction button
# Prediction button
if st.button("Predict Yield"):

    sample = np.array([[state_enc,crop_enc,season_enc,soil_enc,
                        area,rainfall,temperature,
                        humidity,nitrogen,phosphorus,potassium]])

    prediction = model.predict(sample)

    yield_pred = prediction[0]
    production = yield_pred * area

    st.success(f"🌾 Predicted Yield: {yield_pred:.2f} quintal/hectare")
    st.info(f"📦 Estimated Production: {production:.2f} quintal")

    # Yield visualization
    st.subheader("📊 Yield Distribution")

    fig, ax = plt.subplots()

    labels = ["Yield per hectare", "Total Production"]
    values = [yield_pred, production]

    colors = ["#22c55e", "#f97316"]
    explode = (0.1, 0)

    ax.pie(values,
           labels=labels,
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90)

    ax.legend(labels, loc="upper right")
    ax.set_title("Yield vs Production")

    st.pyplot(fig)
