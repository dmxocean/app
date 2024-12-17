import streamlit as st
import pandas as pd
import joblib
import os
import re

def load_models(models_folder="models"):
    models = {}
    if os.path.exists(models_folder):
        model_files = [f for f in os.listdir(models_folder) if f.endswith(".joblib")]
        if not model_files:
            st.warning("No .joblib models found in the 'models' folder.")
        for model_file in model_files:
            model_name = re.sub(r'^(best|best_)?(.*)$', r'\2', model_file.split(".")[0])
            try:
                models[model_name] = joblib.load(os.path.join(models_folder, model_file))
            except Exception as e:
                st.error(f"Failed to load model {model_name}: {str(e)}")
    else:
        st.error(f"Models folder '{models_folder}' does not exist.")
    return models

# Load models dynamically
models = load_models()
if not models:
    st.warning("No models were loaded.")

# Set page title
st.title("Cannabis Use Prediction App")

# Define mapping dictionaries (reversed for natural interface)
mapping_education = {v: k for k, v in {
    -2.43591: "Left school before 16 years",
    -1.73790: "Left school at 16 years",
    -1.43719: "Left school at 17 years",
    -1.22751: "Left school at 18 years",
    -0.61113: "Some college/university, no certificate",
    -0.05921: "Professional certificate/diploma",
    0.45468: "University degree",
    1.16365: "Masters degree",
    1.98437: "Doctorate degree"
}.items()}

mapping_age = {v: k for k, v in {
    -0.95197: "18-24",
    -0.07854: "25-34",
    0.49788: "35-44",
    1.09449: "45-54",
    1.82213: "55-64",
    2.59171: "65+"
}.items()}

mapping_gender = {v: k for k, v in {
    0.48246: "Female",
    -0.48246: "Male"
}.items()}

mapping_countries = {v: k for k, v in {
    0.96082: "UK",
    -0.57009: "USA",
    -0.28519: "Other",
    0.24923: "Canada",
    -0.09765: "Australia",
    0.21128: "Ireland",
    -0.46841: "New Zealand"
}.items()}

mapping_ethnicity = {v: k for k, v in {
    -0.31685: "White",
    0.11440: "Other",
    -1.10702: "Black",
    -0.50212: "Asian",
    0.12600: "Mixed-White/Asian",
    -0.22166: "Mixed-White/Black",
    1.90725: "Mixed-Black/Asian"
}.items()}

# Create two main columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic Information")
    age = st.selectbox("Age", list(mapping_age.keys()))
    gender = st.selectbox("Gender", list(mapping_gender.keys()))
    education = st.selectbox("Education", list(mapping_education.keys()))
    country = st.selectbox("Country", list(mapping_countries.keys()))
    ethnicity = st.selectbox("Ethnicity", list(mapping_ethnicity.keys()))

with col2:
    st.subheader("Personality Scores")
    nscore = st.slider("Neuroticism Score", -3.0, 3.0, 0.0)
    escore = st.slider("Extraversion Score", -3.0, 3.0, 0.0)
    oscore = st.slider("Openness Score", -3.0, 3.0, 0.0)
    ascore = st.slider("Agreeableness Score", -3.0, 3.0, 0.0)
    cscore = st.slider("Conscientiousness Score", -3.0, 3.0, 0.0)
    impuslive = st.slider("Impulsiveness", -3.0, 3.0, 0.0)
    ss = st.slider("Sensation Seeking", -3.0, 3.0, 0.0)

# Define the prediction function
def predict_cannabis_use(input_data, model):
    try:
        # Map categorical inputs to their quantified values
        input_data["age"] = mapping_age.get(input_data["age"], 0)
        input_data["gender"] = mapping_gender.get(input_data["gender"], 0)
        input_data["education"] = mapping_education.get(input_data["education"], 0)
        input_data["country"] = mapping_countries.get(input_data["country"], 0)
        input_data["ethnicity"] = mapping_ethnicity.get(input_data["ethnicity"], 0)

        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return prediction, probability
    
    except KeyError as e:
        st.error(f"Error: Invalid input for {str(e)}.")
        return None, None
    
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {str(e)}")
        return None, None

# Create a prediction button
if st.button("Predict Cannabis Use"):
    
    input_data = {
        "age": age,
        "gender": gender,
        "education": education,
        "country": country,
        "ethnicity": ethnicity,
        "nscore": nscore,
        "escore": escore,
        "oscore": oscore,
        "ascore": ascore,
        "cscore": cscore,
        "impuslive": impuslive,
        "ss": ss
    }
    
    st.subheader("Prediction Results")
    
    # Create a grid layout for model results
    num_cols = len(models)   # Adjust number of columns based on available models

    if num_cols > 0:
        cols = st.columns(num_cols)
        
        for col_index in range(num_cols):
            if col_index < len(models):
                model_name = list(models.keys())[col_index]
                model = models[model_name]
                prediction, probability = predict_cannabis_use(input_data, model)

                with cols[col_index]:
                    st.write(f"**{model_name}**")
                    if prediction is not None and probability is not None:
                        if prediction == 1:
                            st.write("Likely to use cannabis", unsafe_allow_html=True)
                        else:
                            st.write("Unlikely to use cannabis", unsafe_allow_html=True)
                        st.write(f"Probability: {probability:.2f}")
                    else:
                        st.write("Unable to make prediction")
