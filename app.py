import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")
    data["Heart Disease"] = data["Heart Disease"].map({"Absence": 0, "Presence": 1})
    return data

# Train ML model
@st.cache_resource
def train_model(data):
    X = data.drop("Heart Disease", axis=1)
    y = data["Heart Disease"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Load and train
df = load_data()
model, scaler = train_model(df)

# Load external CSS if exists
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Title
st.markdown("<h1>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill the form to predict the chance of heart disease.</p>", unsafe_allow_html=True)

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

# Predict
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob > 0.5 else 0

    # Output box
    if prediction == 1:
        st.markdown(f"""
        <div class='result-box'style='border-left-color: #b8f2e6; background-color:#d8f3dc;'>
            <h4 style='color:red;'>⚠️ High Risk of Heart Disease</h4>
            <p style='color:red;'>Probability: {prob*100:.2f}%</p>
            <p style='color:red;'>Please consult a cardiologist immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-box' style='border-left-color: #b8f2e6; background-color:#d8f3dc;'>
            <h4 style='color:#2a9d8f;'>✅ Low Risk of Heart Disease</h4>
            <p>Probability: {prob*100:.2f}%</p>
            <p>Keep maintaining a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show bar graph after low-risk message
features = ["Age", "Resting BP", "Cholesterol", "Max HR", "ST Depression"]
values = [age, trestbps, chol, thalach, oldpeak]
thresholds = {"Age": 60, "Resting BP": 130, "Cholesterol": 240, "Max HR": 100, "ST Depression": 2.0}

colors = []
for i, val in enumerate(values):
    feat = features[i]
    if (feat == "Max HR" and val < thresholds[feat]) or (feat != "Max HR" and val > thresholds[feat]):
        colors.append("red")
    else:
        colors.append("green")

# Compact bar chart
fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(features, values, color=colors)

ax.set_title("Your Key Health Metrics", fontsize=12)
ax.set_ylim(0, max(values) * 1.2)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height - 10),
                ha="center", va="bottom", color="black" if height > 30 else "black")

st.pyplot(fig)


"""import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("heart.csv")
    data["Heart Disease"] = data["Heart Disease"].map({"Absence": 0, "Presence": 1})  # Convert to numeric
    return data

# Train ML model
@st.cache_resource
def train_model(data):
    X = data.drop("Heart Disease", axis=1)
    y = data["Heart Disease"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced")  # Handle class imbalance
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Load and train
df = load_data()
model, scaler = train_model(df)

# Optional CSS
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Title
st.markdown("<h1>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill the form to predict the chance of heart disease.</p>", unsafe_allow_html=True)

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    # Get probability of class 1 (heart disease)
    probs = model.predict_proba(input_scaled)[0]
    prob = probs[1]  # Probability of heart disease
    prediction = 1 if prob > 0.5 else 0

    # Debug output
   

    # Output
    if prediction == 1:
        st.markdown(f"""
        <div class='result-box'>
            <h4 style='color:#b30000;'>⚠️ High Risk of Heart Disease</h4>
            <p>Probability: {prob*100:.2f}%</p>
            <p>Please consult a cardiologist immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-box' style='border-left-color: #b8f2e6; background-color:#d8f3dc;'>
            <h4 style='color:#2a9d8f;'>✅ Low Risk of Heart Disease</h4>
            <p>Probability: {prob*100:.2f}%</p>
            <p>Keep maintaining a healthy lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)
"""
