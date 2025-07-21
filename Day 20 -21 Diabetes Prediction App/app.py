# ==== Basic Python Libraries Import kar rahe hain ====
import streamlit as st                 # Streamlit: Web app banane ke liye use hoti hai
import joblib                         # joblib: Trained ML model ko load karne ke liye
import numpy as np                    # NumPy: Numerical arrays handle karne ke liye
import pandas as pd                   # Pandas: DataFrame ke form me data manage karne ke liye
import matplotlib.pyplot as plt       # matplotlib: Graphs banane ke liye
import seaborn as sns                 # seaborn: Stylish graphs and plots ke liye

# ==== Trained machine learning model ko load kar rahe hain ====
model = joblib.load('diabetes_model.pkl')  # pkl file me saved trained model ko load kiya

# ==== Streamlit app ka layout design set karte hain ====
st.set_page_config(
    page_title="Diabetes Prediction App",  # App ka tab title
    page_icon="🩺",                        # App ka emoji icon
    layout="wide"                         # Layout ko full screen width me kar diya
)

# ==== Thoda custom CSS styling kar rahe hain UI ko improve karne ke liye ====
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        padding: 20px;
        border-radius: 12px;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==== App ka Title aur basic explanation ====
st.title("🩺 Diabetes Prediction App")  # Page ka main heading
st.markdown("""
This app predicts the likelihood of diabetes using the PIMA Indian Diabetes dataset. 
The model is trained on diagnostic measurements including glucose levels, BMI, age, and other health indicators.
""")  # Description dikhate hain taaki user ko pata chale app kya karta hai

# ==== Sidebar banate hain jaha se user input de sakta hai ====
st.sidebar.header("Patient Parameters")  # Sidebar ka heading
st.sidebar.markdown("Adjust the sliders to input patient health metrics")  # Subheading

# ==== Sliders ke through user input le rahe hain health parameters ====
pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)                # Kitni baar pregnancy hui
glucose = st.sidebar.slider('Glucose Level (mg/dL)', 0, 200, 120)      # Blood glucose level
blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 130, 70) # Blood pressure level
skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)  # Skin thickness
insulin = st.sidebar.slider('Insulin Level (mu U/ml)', 0, 850, 80)     # Insulin level
bmi = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)                         # Body mass index
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.35)  # Diabetes family history score
age = st.sidebar.slider('Age', 20, 90, 30)                             # Patient ki age

# ==== Prediction function: ye model ko inputs deta hai aur result return karta hai ====
def predict():
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]              # 0: No diabetes, 1: Diabetes
    probabilities = model.predict_proba(input_data)[0]     # Dono classes ki probabilities
    return prediction, probabilities

# ==== Page ko 2 columns me divide karte hain: Left = Input Summary, Right = Graphs ====
col1, col2 = st.columns([1, 2])

# ==== LEFT COLUMN: User input ka summary table dikhate hain ====
with col1:
    st.header("Patient Input Summary")
    input_df = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
        'Value': [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    })
    st.table(input_df)  # Table ke form me input values dikhate hain

    # ==== Jab user Predict button dabaye, tab model run kare ====
    if st.button('🔍 Predict Diabetes Risk', use_container_width=True):
        prediction, probabilities = predict()  # prediction aur probabilities nikalte hain
        non_diabetic_prob = probabilities[0] * 100  # No diabetes ki probability
        diabetic_prob = probabilities[1] * 100      # Diabetes hone ki probability

        st.subheader("Prediction Result")

        # ==== Agar prediction diabetes dikhata hai ====
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box diabetic">
                <p class="big-font">High Risk of Diabetes</p>
                <p>Probability: {diabetic_prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Consult a healthcare professional for further evaluation")  # Alert dikhate hain
        else:
            # ==== Agar prediction normal dikhata hai ====
            st.markdown(f"""
            <div class="result-box non-diabetic">
                <p class="big-font">Low Risk of Diabetes</p>
                <p>Probability: {non_diabetic_prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("Maintain healthy lifestyle habits")  # Healthy message show karte hain

# ==== RIGHT COLUMN: Model ke insights aur graphs show karte hain ====
with col2:
    st.header("Model Insights")

    # ==== Feature Importance: kaunsa feature decision me important hai ====
    st.subheader("Feature Importance")
    if hasattr(model, 'coef_'):  # Agar model Logistic Regression hai
        importance = pd.Series(np.abs(model.coef_[0]), index=input_df['Feature'])
    else:  # Agar RandomForest ya tree-based model hai
        importance = pd.Series(model.feature_importances_, index=input_df['Feature'])

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind='barh', ax=ax, color='skyblue')  # Horizontal bar graph
    plt.title('Factors Influencing Diabetes Prediction')
    plt.xlabel('Importance Score')
    st.pyplot(fig)  # Streamlit pe graph dikhate hain

    # ==== Glucose vs BMI ka scatter plot (red lines patient data show karega) ====
    st.subheader("Glucose vs BMI Analysis")
    fig2 = plt.figure(figsize=(10, 6))

    # ==== Dataset ko dobara load karke visual plot banate hain ====
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=names)

    # ==== Scatter plot banate hain jaha x=Glucose, y=BMI, color = Outcome (0/1) ====
    sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data, alpha=0.7)
    plt.axvline(x=glucose, color='r', linestyle='--')  # Patient ka glucose value
    plt.axhline(y=bmi, color='r', linestyle='--')      # Patient ka BMI value
    plt.title('Glucose vs BMI with Patient Position')
    st.pyplot(fig2)

# ==== Footer Section: Credits, accuracy info etc. ====
st.markdown("---")
st.caption("Model trained on PIMA Indian Diabetes Dataset | Accuracy: 77-80%")

# ==== End of code ====
# streamlit run app.py
