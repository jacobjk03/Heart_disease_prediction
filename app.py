import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.write(""" # Heart Disease Prediction App
This app predicts that *the person has heart disease or not* 
""")

def user_input_features():

    with st.sidebar:
    
        st.write("""**1. Select Age :**""") 
        age = st.slider('', 0, 100, 25)
        st.write("""**You selected this option **""",age)
    
        st.write("""**2. Select Gender :**""")
        sex = st.selectbox("(1=Male, 0=Female)",["1","0"])
        st.write("""**You selected this option **""",sex)
    
        st.write("""**3. Select Chest Pain Type :**""")
        cp = st.selectbox("(0 = Typical Angina, 1 = Atypical Angina, 2 = Non—anginal Pain, 3 = Asymptotic) : ",["0","1","2","3"])
        st.write("""**You selected this option **""",cp)
    
        st.write("""**4. Select Resting Blood Pressure :**""")
        trestbps = st.slider('In mm/Hg unit', 0, 200, 110)
        st.write("""**You selected this option **""",trestbps)
    
        st.write("""**5. Select Serum Cholesterol :**""")
        chol = st.slider('In mg/dl unit', 0, 600, 115)
        st.write("""**You selected this option **""",chol)

        st.write("""**2. Select Fasting Sugar :**""")
        fbs = st.selectbox("(1=True, 0=False)",["1","0"])
        st.write("""**You selected this option **""",fbs)

        st.write("""**2. Select Resting Electocardiographic Results :**""")
        restecg = st.selectbox("(0=Nothing to Note, 1: ST-T Wave abnormality, 2: Possible or definite left ventricular hypertrophy)",["0","1", "2"])
        st.write("""**You selected this option **""",restecg)
    
        st.write("""**6. Maximum Heart Rate Achieved (THALACH) :**""")
        thalach = st.slider('', 0, 220, 115)
        st.write("""**You selected this option **""",thalach)
    
        st.write("""**7. Exercise Induced Angina (Pain in chest while exersice) :**""")
        exang = st.selectbox("(1=Yes, 0=No)",["1","0"])
        st.write("""**You selected this option **""",exang)
    
        st.write("""**8. Oldpeak (ST depression induced by exercise relative to rest) :**""")
        oldpeak = float(st.slider('', 0.0, 10.0, 2.0))
        st.write("""**You selected this option **""",oldpeak)
    
        st.write("""**9. Slope (The slope of the peak exercise ST segment) :**""")
        slope = st.selectbox("(Select 0, 1 or 2)",["0","1","2"])
        st.write("""**You selected this option **""",slope)
    
        st.write("""**10. CA (Number of major vessels (0-3) colored by flourosopy) :**""")
        ca = st.selectbox("(Select 0, 1, 2 or 3)",["0","1","2","3"])
        st.write("""**You selected this option **""",ca)
    
        st.write("""**11. Thal :**""")
        thal = float(st.slider('3 = normal; 6 = fixed defect; 7 = reversable defect', 0.0, 10.0, 3.0))
        st.write("""**You selected this option **""",thal)
    
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('Given Inputs : ')
st.write(df)

dataset = pd.read_csv("heart-disease.csv")
dataset.isna()

# Split the data int X and y
X = dataset.drop("target", axis=1)
y = dataset["target"]

# Split data into train and test
np.random.seed(42)

#Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train, y_train)

prediction_proba = clf.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba*100)

prediction = clf.predict(df)
st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'REPORT'] = 'According to the model *NO HEART DISEASE DETECTED*'
df1.loc[df1['0'] == 1, 'REPORT'] = 'According to the model *HEART DISEASE DETECTED*'
st.write(df1)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
