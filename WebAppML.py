from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diabetes Detection",page_icon="./Imagenes/learning.png")
st.write("""
# Diabetes Detection 
Detect if someone has diabetes using machine learning and python
""")
image = Image.open('./Imagenes/imagen.png')
st.image(image,caption="ML",use_column_width=True)
#Obtenemos los datos/home/anon23/Escritorio/WebAppML/imagen.png
df = pd.read_csv('./Data/diabetes.csv')

#Set WebApp subheader
st.subheader('Data information')
#Show data as datatable
st.dataframe(df)
#Show statitstics on the data
st.write(df.describe())
#Show the data as a chart
chart = st.bar_chart(df)
#Split the data into independent 'X' and independent 'Y' variables
X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values
#Split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state=0)
#Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    Glucose = st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('Blood_Pressure',0,122,72)
    SkinThickness =  st.sidebar.slider('SkinThickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0.0,846.0,30.5)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    Age  = st.sidebar.slider('Age',21,81,29)

    #Store a dictionary into a variable
    user_data = {'pregnancies' : pregnancies,
                 'glucose':Glucose,
                 'blood_pressure':BloodPressure,
                 'skin_thikness':SkinThickness,
                 'insulin':Insulin,
                 'BMI':BMI,
                 'DPF':DPF,
                 'Age':Age
                }
    #Transform the data into a data frame 
    features = pd.DataFrame(user_data,index = [0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User input')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test)* 100)) + '%')
#Store the models predictions in a variable
prediction =RandomForestClassifier.predict(user_input)
#Set a subheader and display the clasification
st.subheader('Clasification')
st.write(prediction)


