
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = None
try:
    with open("Expenses_Predictor.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚫 Model file 'Expenses_Predictor.pkl' not found. Please ensure it is located in the application directory.")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred while loading the model: {e}")

# App title
st.title("💊 Medical Expenses Predictor")

if model:
    st.header("🧑‍⚕️ Patient Information")

    # Categorical mappings
sex_mapping = {"female": 0, "male": 1}
smoker_mapping = {"no": 0, "yes": 1}
region_mapping = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}

    # Input fields
age = st.slider("Age", 18, 65, 30)
sex_encoded=st.selectbox("Sex", list(sex_mapping.keys()))
bmi = st.slider("BMI", 15.0, 50.0, 25.0, help="Body Mass Index: normal range is 18.5 - 24.9")
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", list(smoker_mapping.keys()))
region = st.selectbox("Region", list(region_mapping.keys()))

    # Predict button
if st.button("Predict Medical Expenses"):
      

        # Prepare input data
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex_encoded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        # Make prediction
        try:
            prediction = model.predict(input_data)
            st.success("✅ Prediction completed successfully!")
            st.header("💰 Estimated Medical Expenses")
            st.write(f"### ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

st.markdown("---")
st.info("📊 Note: This model was trained on an insurance dataset to estimate medical expenses based on patient attributes.")

