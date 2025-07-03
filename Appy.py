import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
model = None
try:
    with open("Expenses_Predictor.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# App UI
st.title("💊 Medical Expenses Predictor")

if model:
    st.header("🧑‍⚕️ Enter Patient Information")

    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict Expenses"):
        # Encode input like in training
        sex_encoded = 1 if sex == "male" else 0
        smoker_encoded = 1 if smoker == "yes" else 0
        region_encoded = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex_encoded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker_encoded,
            'region': region_encoded
        }])

        # Predict
        prediction = model.predict(input_df)
        st.success("✅ Prediction successful!")
        st.write(f"### 💰 Predicted Medical Expenses: ${prediction[0]:,.2f}")
else:
    st.warning("⚠️ Model could not be loaded. Please check the file.")
