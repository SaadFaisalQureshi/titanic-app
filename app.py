"""
Heart Disease Prediction System - Streamlit Web App
----------------------------------------------------
Purpose of this file:
This small web app lets anyone use the trained Support Vector Classifier
(SVC) from 'Heart_Disease_Prediction.ipynb' through a browser form.

How it works:
1. Loads the trained model 'heart_svc_trained_model.pkl' made by the notebook.
2. Re-creates the same LabelEncoders used in training (same category lists),
   so text inputs are converted to the same numeric codes.
3. Collects the 4 inputs with dropdowns, encodes them, and shows the prediction.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Basic page settings (browser tab title, icon and layout)
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Constants: model file path and the possible values of each input attribute
# ---------------------------------------------------------------------------

# The model file must be in the same folder as this app.py
MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"

# These lists MUST match the ones used in the notebook (Step 5.1)
SEX = ["Male", "Female"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
REST_ECG = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
EXERCISE_ANGINA = ["Yes", "No"]


# ---------------------------------------------------------------------------
# Loading the model and encoders (cached so they load only once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained SVC model from the .pkl file."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """
    Re-create the notebook's LabelEncoders.
    A LabelEncoder's codes depend only on the sorted category list, so
    fitting on the same lists gives exactly the same codes as in training.
    """
    return {
        "Sex": LabelEncoder().fit(SEX),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "RestECG": LabelEncoder().fit(REST_ECG),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
    }


model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content: title, description and input form
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter the four clinical attributes below and the trained Support Vector
    Classifier (SVC) will predict whether heart disease is likely.
    Trained on the UCI Heart Disease (Cleveland) dataset.

    *Educational demo only. This is not medical advice or a diagnostic tool.*
    """
)
st.divider()

# Two columns of dropdowns (Step 8.1: take input from the user)
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
with col2:
    rest_ecg = st.selectbox("Resting ECG Result", REST_ECG)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

# ---------------------------------------------------------------------------
# Prediction (runs only when the button is clicked)
# ---------------------------------------------------------------------------

if st.button("Predict", type="primary", use_container_width=True):

    # Step 8.2: put the user's input into a one-row DataFrame
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "RestECG": [rest_ecg],
        "ExerciseAngina": [exercise_angina],
    })

    # Step 8.3: label encode each column with its own encoder
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col])
                            for col in user_input.columns})

    # Step 8.5: predict (the model returns an array like [1], so take [0])
    prediction = model.predict(encoded)[0]

    # Show the result: 1 = heart disease likely, 0 = not likely
    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    # Optional: show the numbers actually sent to the model
    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Built for AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus.")
