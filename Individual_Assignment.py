import streamlit as st
import pickle
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Function to load the XGBoost model
def load_xgboost_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'assignment_model_2_final.pkl')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

# Function to load the RandomForest model
def load_randomforest_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'Potability_Model.pkl')
    model = joblib.load(model_path)
    return model

# Load models in the global scope
xgboost_model = load_xgboost_model()
randomforest_model = load_randomforest_model()


# Streamlit app
def main():
    st.title("ML Model Showcase")

    # Home screen to select the model
    model_choice = st.sidebar.radio("Select Model", ("XGBoost", "RandomForest"))

    if model_choice == "XGBoost":
        st.header("XGBoost Model")

        def preprocess_data(data):
            # Features to be scaled
            features_to_scale = ['Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Solidity', 'Extent', 'Roundness', 'Aspect_Ratio', 'Compactness']

            # Apply standard scaling to the selected features
            scaler = StandardScaler()
            data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

            return data

        # Function to make predictions
        # Function to make predictions
        def predict(data):
            # Preprocess the input data
            data_processed = preprocess_data(data)

            # Convert pandas DataFrame to DMatrix
            dmatrix = xgb.DMatrix(data_processed)

            # Make predictions
            predictions = xgboost_model.predict(dmatrix)

            return predictions

        average_major_axis_length = 456.60
        average_minor_axis_length = 225.79
        average_eccentricity = 0.86
        average_solidity = 0.99
        average_extent = 0.69
        average_roundness = 0.79
        average_aspect_ratio = 2.04
        average_compactness = 0.70
        # Input form with input fields

        major_axis_length = st.number_input('Major Axis Length', min_value=0.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Major Axis Length is {:.2f}</p>'.format(average_major_axis_length), unsafe_allow_html=True)

        minor_axis_length = st.number_input('Minor Axis Length', min_value=0.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Minor Axis Length is {:.2f}</p>'.format(average_minor_axis_length), unsafe_allow_html=True)

        eccentricity = st.number_input('Eccentricity', min_value=0.0, max_value=1.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Eccentricity is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_eccentricity), unsafe_allow_html=True)

        solidity = st.number_input('Solidity', min_value=0.0, max_value=1.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Solidity is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_solidity), unsafe_allow_html=True)

        extent = st.number_input('Extent', min_value=0.0, max_value=1.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Extent is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_extent), unsafe_allow_html=True)

        roundness = st.number_input('Roundness', min_value=0.0, max_value=1.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Roundness is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_roundness), unsafe_allow_html=True)

        aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Aspect Ratio is {:.2f}</p>'.format(average_aspect_ratio), unsafe_allow_html=True)

        compactness = st.number_input('Compactness', min_value=0.0, max_value=1.0, value=1.0)
        st.markdown('<p style="font-size: smaller; font-style: italic;">The average Compactness is {:.2f}. You can maximally input a value of 1.00.</p>'.format(average_compactness), unsafe_allow_html=True)


        if not major_axis_length or not minor_axis_length or not eccentricity or not solidity or not extent or not roundness or not aspect_ratio or not compactness:
            st.error('Error: All fields are mandatory. Please fill in all measurements.')
        else:
            user_input = pd.DataFrame({
                'Major_Axis_Length': [major_axis_length],
                'Minor_Axis_Length': [minor_axis_length],
                'Eccentricity': [eccentricity],
                'Solidity': [solidity],
                'Extent': [extent],
                'Roundness': [roundness],
                'Aspect_Ratio': [aspect_ratio],
                'Compactness': [compactness]
            })


            if st.button("Predict"):
                result = predict(user_input)

                # Map the prediction to the corresponding class
                class_mapping = {0: 'Çerçevelik', 1: 'Ürgüp Sivrisi'}
                prediction_label = class_mapping[int(result[0])]

                st.write('This pumpkin seed is of quality type', prediction_label + '.')


    elif model_choice == "RandomForest":
        st.header("RandomForest Model")

        def preprocess_data_rf(data):
            data["pHxSulfate"] = data["ph"] * data["Sulfate"]
            data["SulfatexChloramines"] = data["Sulfate"] * data["Chloramines"]
            return data

        # Function to make predictions for RandomForest
        def predict_rf(data):
            # Preprocess the input data
            data_processed = preprocess_data_rf(data)

            # Make predictions
            predictions = randomforest_model.predict(data_processed)

            return predictions


        # Example input features for RandomForest
        # Assuming ph, Hardness, Solids, ... are the column names in your dataset
        ph = st.sidebar.slider('pH', min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.sidebar.slider('Hardness', min_value=0.0, max_value=500.0, value=200.0)
        solids = st.sidebar.slider('Solids', min_value=0.0, max_value=5000.0, value=1000.0)
        chloramines = st.sidebar.slider('Chloramines', min_value=0.0, max_value=10.0, value=4.0)
        sulfate = st.sidebar.slider('Sulfate', min_value=0.0, max_value=500.0, value=100.0)
        conductivity = st.sidebar.slider('Conductivity', min_value=0.0, max_value=2000.0, value=800.0)
        organic_carbon = st.sidebar.slider('Organic Carbon', min_value=0.0, max_value=50.0, value=10.0)
        trihalomethanes = st.sidebar.slider('Trihalomethanes', min_value=0.0, max_value=100.0, value=50.0)
        turbidity = st.sidebar.slider('Turbidity', min_value=0.0, max_value=10.0, value=5.0)

        # Create a dictionary with the input data
        input_data = {
            'ph': ph,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }

        if st.button("Predict"):
                predict_rf(input_data)

                # Map the prediction to the corresponding class
                class_mapping = {0: 'potable', 1: 'not potable'}
                prediction_label = class_mapping[int(result[0])]

                st.write('The water is', prediction_label + '.')


if __name__ == "__main__":
    main()
