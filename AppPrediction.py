import streamlit as st

import pandas as pd
import numpy as np

import os
import joblib

import matplotlib.pyplot as plt
import matplotlib as matplotlib
matplotlib.use('Agg')

st.title("")

# Image
from PIL import Image
image = Image.open('./HR.png')
st.image(image, caption='Predictea GmbH', use_column_width=True)

geography_dict = {"Germany":1, "France":2, "Spain":3}
gender_dict = {"female":0, "male":1}
feature_dict = {"No":0, "Yes":1}

def get_geography_value(val):
    geography_dict = {"Germany":1, "France":2, "Spain":3}
    for key, value in geography_dict.items():
        if val == key:
            return value

def get_gender_value(val):
    gender_dict = {"female":0, "male":1}
    for key, value in gender_dict.items():
        if val == key:
            return value

def get_feature_value(val):
    feature_dict = {"No":0, "Yes":1}
    for key, value in feature_dict.items():
        if val == key:
            return value

def main():
    """Prediction App"""

    menu = ["Home", "Login", "About"]
    submenu = ["Visualisation", "Prediction"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.title("PYTHON APP SOLUTION")
        st.title("Do you want to know your EMPLOYEES?")
        st.subheader("If so, please Log IN!")
        st.write("Special market situations require flexible and efficient solutions, be it for example the successful implementation of a new project, coping with a crisis in a timely manner or covering a sudden vacancy in a key management position. The rapid implementation of measures is crucial to success in today's dynamic economic environment!")
        st.text("")
        st.text("")
        st.text("")
        st.write("Copyright © 2021")

    elif choice == "Login":
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            if password in ["HR", "123"]:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Activity", submenu)
                if activity == "Visualisation":
                    st.subheader("Data Visualisation")
                    df = pd.read_csv("HR_data.csv")
                    st.dataframe(df)

                    fig1 = plt.figure()
                    df['Geography'].value_counts().plot(kind='bar', color='dodgerblue')
                    plt.title('Geography')
                    st.pyplot(fig1)

                    fig2 = plt.figure()
                    df['Gender'].value_counts().plot(kind='bar', color='dodgerblue')
                    plt.title('Gender')
                    st.pyplot(fig2)

                    fig3 = plt.figure()
                    df['Age'].hist(alpha=1,color='dodgerblue', bins=30)
                    plt.title('Age Distribution')
                    st.pyplot(fig3)
                    
                    fig4 = plt.figure()
                    df['Tenure'].hist(alpha=1,color='dodgerblue', bins=11)
                    plt.title('Tenure Distribution')
                    st.pyplot(fig4)

                    fig5 = plt.figure()
                    df['ActiveEmployee'].value_counts().plot(kind='bar', color='dodgerblue')
                    plt.title('Active Employee')
                    st.pyplot(fig5)

                    fig6 = plt.figure()
                    df['Salary'].hist(alpha=1,color='dodgerblue', bins=100)
                    plt.title('Salary Distribution')
                    st.pyplot(fig6)

                    fig7 = plt.figure()
                    df['Left'].value_counts().plot(kind='bar', color='dodgerblue')
                    plt.title('People Left')
                    st.pyplot(fig7)     

                elif activity == "Prediction":
                    st.subheader("Please choose atributes:")

                    geography = st.radio("Geography:", tuple(geography_dict.keys()))
                    gender = st.radio("Gender:", tuple(gender_dict.keys()))
                    age = st.number_input("Age:",1,100,18)
                    tenure = st.number_input("Years at the firm:",1,45,1)
                    active = st.radio("Is active employee:", tuple(feature_dict.keys()))
                    salary = st.slider("Yearly salary:",10000,200000,10000)

                    feature_list = [get_geography_value(geography), get_gender_value(gender), age, tenure, get_feature_value(active), salary]
                    st.subheader("Output to be send to the model:")
                    pretty_result = {"geography": geography, "gender": gender, "age": age, "tenure": tenure, "active": active, "salary": salary}
                    st.json(pretty_result)

                    data = np.array(feature_list).reshape(1,-1)
                    data_sample = pd.DataFrame(data).rename(columns = {0:"Geography", 1:"Gender", 2:"Age", 3:"Tenure", 4:"ActiveEmployee", 5:"Salary"}) 

                    # ML
                    model_choice = st.selectbox("Select model:", ["Logistic regression", "K-NN", "Random forest"])
                    if st.button("Predict"):
                        if model_choice == "Logistic regression":
                            
                            df = pd.read_csv("HR_data.csv")
                            df = df[["Geography","Gender","Age","Tenure","ActiveEmployee","Salary","Left"]]
                            df['Geography'] = df['Geography'].replace(["Germany"], 1)
                            df['Geography'] = df['Geography'].replace(["France"], 2)
                            df['Geography'] = df['Geography'].replace(["Spain"], 3)
                            df['Gender'] = df['Gender'].replace(["Female"], 0)
                            df['Gender'] = df['Gender'].replace(["Male"], 1)

                            X = df.drop('Left', axis=1)
                            y = df['Left']

                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

                            from sklearn.linear_model import LogisticRegression
                            logmodel = LogisticRegression()
                            logmodel.fit(X_train,y_train)
                            predictions = logmodel.predict(X_test)

                            from sklearn.metrics import confusion_matrix
                            cm = confusion_matrix(y_test,predictions)                         
                            proc = round((cm[0,0]+cm[1,1])/((sum(cm)[0]+sum(cm)[1])/100),2)
                            st.write('Accuracy: {}%'.format(proc))

                            predictions = logmodel.predict(data_sample)
                            if predictions == 0:
                                st.success('The employee will stay :)')
                            else:
                                st.warning('The employee will leave :(')
                        
                        elif model_choice == "K-NN":
                            df = pd.read_csv("HR_data.csv")
                            df = df[["Geography","Gender","Age","Tenure","ActiveEmployee","Salary","Left"]]
                            df['Geography'] = df['Geography'].replace(["Germany"], 1)
                            df['Geography'] = df['Geography'].replace(["France"], 2)
                            df['Geography'] = df['Geography'].replace(["Spain"], 3)
                            df['Gender'] = df['Gender'].replace(["Female"], 0)
                            df['Gender'] = df['Gender'].replace(["Male"], 1)

                            X = df.drop('Left', axis=1)
                            y = df['Left']

                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

                            from sklearn.neighbors import KNeighborsClassifier
                            knn = KNeighborsClassifier(n_neighbors=18)
                            knn.fit(X_train,y_train)
                            prediction2 = knn.predict(X_test)

                            from sklearn.metrics import classification_report, confusion_matrix
                            cm = confusion_matrix(y_test,prediction2)
                            proc = round((cm[0,0]+cm[1,1])/((sum(cm)[0]+sum(cm)[1])/100),2)
                            st.write('Accuracy: {}%'.format(proc))

                            predictions = knn.predict(data_sample)
                            if predictions == 0:
                                st.success('The employee will stay :)')
                            else:
                                st.warning('The employee will leave :(')

                        elif model_choice == "Random forest":
                            df = pd.read_csv("HR_data.csv")
                            df = df[["Geography","Gender","Age","Tenure","ActiveEmployee","Salary","Left"]]
                            df['Geography'] = df['Geography'].replace(["Germany"], 1)
                            df['Geography'] = df['Geography'].replace(["France"], 2)
                            df['Geography'] = df['Geography'].replace(["Spain"], 3)
                            df['Gender'] = df['Gender'].replace(["Female"], 0)
                            df['Gender'] = df['Gender'].replace(["Male"], 1)
                            
                            X = df.drop('Left', axis=1)
                            y = df['Left']

                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

                            from sklearn.ensemble import RandomForestClassifier
                            rfc = RandomForestClassifier(n_estimators=600)
                            rfc.fit(X_train,y_train)
                            prediction3 = rfc.predict(X_test)

                            from sklearn.metrics import classification_report, confusion_matrix
                            cm = confusion_matrix(y_test,prediction3)
                            proc = round((cm[0,0]+cm[1,1])/((sum(cm)[0]+sum(cm)[1])/100),2)
                            st.write('Accuracy: {}%'.format(proc))

                            predictions = rfc.predict(data_sample)
                            if predictions == 0:
                                st.success('The employee will stay :)')
                            else:
                                st.warning('The employee will leave :(')

            else:
                st.warning("Incorrect Username or Password!")
   
    elif choice == "About":
        st.title("About me / My mission:")
        st.subheader("I can help you to predict, if and when your employees will leave your company!")
        st.write("My name is Jaroslav Kotrba and I am very passionate about working with data and its visualization or transformation in an understandable manner. In year 2020 I wrote and defended my diploma thesis: 'The Systemic Risk at the Mortgage Market', which allowed me to unfold my skills in the data science area."
                 "My passion for numbers, modeling, and predictions led me to courses linked to econometrics and statistics taught at the Faculty of Economics, where I got acquainted with data analytics. Then followed courses related to the statistics in computer science at the Faculty of Informatics that were not compulsory. However, I wanted to get a better insight and acquire more knowledge in practical data science from real IT experts."
                 "The first working experience was obtained at the Czech Embassy in Vienna. My tasks were to compile statistics related to the Austrian economy and write summarisations on them accompanied by graphic elements predominantly sent to the headquarters in Prague. Writing about Prague gives me the opportunity to mention the membership in the Economics Commission for National Economy, where I discuss and elaborate possible changes, steps, and recommendations for the government to stabilize and improve the Czech Republic's economic situation. Nowadays, I devote my time to work with software R where I love to solve new problems and challenges. As I do like R very much, I am very keen on dealing with SQL, Python, Excel, or Tableau. I am currently working by Honeywell as Data Analyst.")

if __name__ == '__main__':
    main()
