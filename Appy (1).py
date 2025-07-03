import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = None
try:
    with open("Expenses_Predictor.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚫 Model file 'Expenses_Predictor.pkl' not found in the current directory.")
except Exception as e:
    st.error(f"⚠️ An error occurred while loading the model:

{e}")

# App title
st.title("💊 Medical Expenses Predictor")

if model:
    st.header("🧑‍⚕️ Enter Patient Details")

    # Inputs
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    # On button click
    if st.button("Predict Medical Expenses"):
        # Encoding inputs
        sex_encoded = 1 if sex == "male" else 0
        smoker_encoded = 1 if smoker == "yes" else 0
        region_encoded = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}[region]

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex_encoded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker_encoded,
            'region': region_encoded
        }])

        # Make prediction
        try:
            prediction = model.predict(input_df)
            st.success("✅ Prediction successful!")
            st.write(f"### 💰 Estimated Expenses: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("⚠️ Model not loaded. Please check the file.")