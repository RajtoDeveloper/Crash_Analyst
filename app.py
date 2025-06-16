import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the app
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

# Title and description
st.title("🚗 Accident Severity Prediction Dashboard")
st.markdown("Predict and visualize accident severity based on key factors.")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("accident_severity_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data["xgboost_model"], {0: "Slight", 1: "Serious", 2: "Fatal"}
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

model, severity_mapping = load_model()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Light', 'Casualties', 'Vehicles', 'Road', 'Speed', 
        'Area', 'Vehicle_Type', 'Weather', 'Prediction'
    ])

# Input form
with st.expander("🔧 Input Parameters", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        light = st.selectbox("Light Conditions", 
                           ["Daylight", "Dark - Lights On", "Dark - No Lights"])
        casualties = st.number_input("Number of Casualties", min_value=0, max_value=20, value=1)
        vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=10, value=2)
        road = st.selectbox("Road Surface", ["Dry", "Wet", "Snow/Ice"])
        
    with col2:
        speed = st.slider("Speed Limit (km/h)", min_value=20, max_value=120, value=50)
        area = st.radio("Area Type", ["Urban", "Rural"])
        vehicle_type = st.selectbox("Vehicle Type", ["Car", "Motorcycle", "Bus", "Truck"])
        weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Snow/Fog"])
    
    submitted = st.button("Predict Severity")

# Prediction and Visualization
if submitted and model:
    try:
        # Convert inputs
        input_mapping = {
            "light": {"Daylight": 0, "Dark - Lights On": 1, "Dark - No Lights": 2},
            "road": {"Dry": 0, "Wet": 1, "Snow/Ice": 2},
            "area": {"Urban": 1, "Rural": 0},
            "vehicle": {"Car": 0, "Motorcycle": 1, "Bus": 2, "Truck": 3},
            "weather": {"Clear": 1, "Rain": 2, "Snow/Fog": 3}
        }
        
        features = [
            input_mapping["light"][light],
            casualties,
            vehicles,
            input_mapping["road"][road],
            speed,
            input_mapping["area"][area],
            input_mapping["vehicle"][vehicle_type],
            input_mapping["weather"][weather],
            0, 2, 0, 1, 0  # Defaults
        ]
        
        # Make prediction
        severity = model.predict(np.array(features).reshape(1, -1))[0]
        severity_text = severity_mapping.get(severity, "Unknown")
        
        # Add to history
        new_entry = pd.DataFrame([{
            'Light': light,
            'Casualties': casualties,
            'Vehicles': vehicles,
            'Road': road,
            'Speed': speed,
            'Area': area,
            'Vehicle_Type': vehicle_type,
            'Weather': weather,
            'Prediction': severity_text
        }])
        st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
        
        # Visualization columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution pie chart
            st.subheader("Severity Distribution")
            if len(st.session_state.history) > 0:
                fig1, ax1 = plt.subplots()
                counts = st.session_state.history['Prediction'].value_counts()
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
                       colors=['#4CAF50', '#FFC107', '#F44336'])
                ax1.axis('equal')
                st.pyplot(fig1)
            else:
                st.info("No prediction history yet")
            
        with col2:
            # Feature importance bar chart
            st.subheader("Key Influencing Factors")
            if hasattr(model, 'feature_importances_'):
                features = [
                    'Light', 'Casualties', 'Vehicles', 'Road', 'Speed',
                    'Area', 'Vehicle', 'Weather', 'RoadType', 'Junction',
                    'Pedestrian', 'Special', 'Hazards'
                ]
                importance = model.feature_importances_
                fig2, ax2 = plt.subplots()
                sns.barplot(x=importance[:8], y=features[:8], palette="Blues_d", ax=ax2)
                ax2.set_xlabel("Importance Score")
                st.pyplot(fig2)
            else:
                st.info("Feature importance not available")
        
        # Show latest prediction result
        st.subheader(f"Prediction Result: {severity_text}")
        if severity_text == "Slight":
            st.success("This accident likely results in minor injuries")
        elif severity_text == "Serious":
            st.warning("This accident may cause significant injuries requiring hospitalization")
        else:
            st.error("This accident has high risk of fatalities")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Show historical predictions
if not st.session_state.history.empty:
    st.subheader("📊 Prediction History")
    st.dataframe(st.session_state.history)
    
    # Historical trends chart
    st.subheader("Trend Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Casualties vs Severity")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=st.session_state.history, x='Prediction', y='Casualties', 
                   order=["Slight", "Serious", "Fatal"],
                   palette=["#4CAF50", "#FFC107", "#F44336"])
        st.pyplot(fig3)
    
    with col2:
        st.write("Speed Distribution by Severity")
        fig4, ax4 = plt.subplots()
        sns.violinplot(data=st.session_state.history, x='Prediction', y='Speed',
                      order=["Slight", "Serious", "Fatal"],
                      palette=["#4CAF50", "#FFC107", "#F44336"])
        st.pyplot(fig4)

# Add footer
st.markdown("---")
st.caption("Accident Severity Prediction Dashboard | Made with Streamlit")
