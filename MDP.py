import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,mean_absolute_error,mean_squared_error 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor ,VotingClassifier
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from sklearn.svm import SVC
from streamlit_option_menu import option_menu
ros=RandomOverSampler()
scaler=MinMaxScaler((-1,1))
scaler1=MinMaxScaler((-6,1))
scaler2=MinMaxScaler((-0.2,1))

with st.sidebar:
    selected=option_menu(
        menu_title='Multiple Disease Prediction',
        options=['Parkinsons Predictions','Liver Prediction','Kidney Prediction'],
        icons=["house-door",  "graph-up", "file-earmark-zip","book-half"],
        default_index=0,


    )   

def handle_outliers_iqr(rd, cols):
    for col in cols:
        Q1 = rd[col].quantile(0.25)
        Q3 = rd[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        rd[col] = np.where(rd[col] < lower_bound, lower_bound, rd[col])
        rd[col] = np.where(rd[col] > upper_bound, upper_bound, rd[col])
    return rd

if selected=='Parkinsons Predictions':
    col1,col2=st.columns(2)
    with col1:
     st.title(":red[PARKINSON'S DISEASE PREDICTION]")   
    with col2:
     st.image(r'https://th.bing.com/th/id/OIP.FrbaFXWHa1A_ud4-xPlfNwHaL4?w=156&h=194&c=7&r=0&o=5&dpr=1.3&pid=1.7')
    df=pd.read_csv(r"C:\Users\firea\Downloads\parkinsons.csv")
    with open('parkinson.pkl','wb') as file:
        pickle.dump(df,file)
    with open('parkinson.pkl','rb') as file:
        df=pickle.load(file)
    #SELECTING TARGET AND FEATURE COLUMN
    feature=df.drop(['status','name'],axis=1)
    target=df['status']
    #balancing the data using randomoversampling
    feature_sampled,target_sampled=ros.fit_resample(feature,target)
    #FEATURE SCALING THE FEATURE COLUMN
    feature_x=scaler.fit_transform(feature_sampled)
    target_y=target_sampled
    

#PREPROCESSING COMPLETED..

    #TRAIN_TEST_SPLIT
    feature_train,feature_test,target_train,target_test=train_test_split(feature_x,target_y)
    #MODEL1-LOGISTIC REGRESSION
    LR=LogisticRegression(C=0.4,max_iter=1000,solver='liblinear')
  

    #MODEL2-DECISIONTREECLASSIFIER
    DTR=DecisionTreeClassifier()

    #MODEL3-XGBOOST


    #MODEL4-KNN
    KNN=KNeighborsClassifier()
   

    #MODEL5-RANDOMFORESTCLASSIFIER
    RFC=RandomForestClassifier(criterion='entropy',random_state=14)
   
    #MODEL6-SVM
    SVM=SVC(probability=True)

    

    #VOTINGCLASSIFIER
    EVC=VotingClassifier(estimators=[('LR',LR),('DTR',DTR),('KNN',KNN),('RFC',RFC),('SVM',SVM)],voting='hard')
    EVC.fit(feature_train,target_train)
    #PREDICTION
    target_predevc=EVC.predict(feature_test)
    
    #ACCURACY
    acc_evc=accuracy_score(target_test,target_predevc)
    name=st.text_input('Patient name:')
    col1,col2,col3,col4,col5=st.columns(5)
    
    
    

    with col1:
    #    st.header('MDVP:Fo(HZ)')
       A=st.number_input(label='MDVP:Fo(Hz)')
       B=st.number_input(label='MDVP:Fhi(Hz)')
       C=st.number_input(label='MDVP:Flo(Hz)')
       D=st.number_input(label='MDVP:Jitter(%)')
       
       E=st.number_input(label='MDVP:Jitter(Abs)')
       
       
    with col2:
       F=st.number_input(label='MDVP:RAP')
       G=st.number_input(label='MDVP:PPQ')
       H=st.number_input(label='Jitter:DDP')
       I=st.number_input(label='MDVP:Shimmer')
       J=st.number_input(label='MDVP:Shimmer(dB)')
       

    with col3:
       K=st.number_input(label='Shimmer:APQ3')
       L=st.number_input(label='Shimmer:APQ5')
       M=st.number_input(label='MDVP:APQ')
       N=st.number_input(label='Shimmer:DDA')
       O=st.number_input(label='NHR')

    with col4:
       
       Q=st.number_input(label='HNR')
       R=st.number_input(label='RPDE')
       S=st.number_input(label='DFA')
       T=st.number_input(label='spread1')
       P=st.number_input(label='spread2')
    
    with col5:
       U=st.number_input(label='D2')
       V=st.number_input(label='PPE')
       
    input_data=np.array([[A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V]])
       
    if st.button('Predict'):
     predict=EVC.predict(input_data)
     if predict==1:
        with st.spinner('Predicting'):
         st.error(f"Patient {name} has Parkinson's disease")
     else:
        with st.spinner('Prediction'):
         st.success(f"Patient {name} does not have parkinson's disease")
     
    
if selected=='Kidney Prediction':
   col1,col2=st.columns(2)
   with col1:
    st.title(':red[KIDNEY DISEASE PREDICTION]')   
   with col2:
    st.image(r'https://th.bing.com/th/id/OIP.vxeXslFQ0Rr0TYGPlcWfxgHaLb?w=156&h=195&c=7&r=0&o=5&dpr=1.3&pid=1.7')
   df1=pd.read_csv(r"C:\Users\firea\Downloads\Kidney.csv")
   feature=df1[['sc','bu','hemo','al','bp']]
   target1=df1['classification']
   colls=feature.columns
   #HANDLING OULTILERS
   feature_handled=handle_outliers_iqr(feature,colls)
   #HANDLING NULL VALUES WITH MEAN
   for i in feature_handled.columns:
      feature_handled[i].fillna(feature_handled[i].mean(),inplace=True)
      
    #BALANCING THE DATA 
   feature_sampled1,target_y1=ros.fit_resample(feature_handled,target1)

   #FEATURE SCALING
   feature_x1=scaler1.fit_transform(feature_sampled1)
#PREPROCESSING COMPLETED..

    #getting train,test samples
   feature_train1,feature_test1,target_train1,target_test1=train_test_split(feature_x1,target_y1)
       #MODEL1-LOGISTIC REGRESSION
   LR=LogisticRegression(C=0.4,max_iter=1000,solver='liblinear')
  

    #MODEL2-DECISIONTREECLASSIFIER
   DTR=DecisionTreeClassifier()

    #MODEL4-KNN
   KNN=KNeighborsClassifier()
   

    #MODEL5-RANDOMFORESTCLASSIFIER
   RFC=RandomForestClassifier(criterion='entropy',random_state=14)
   
    #MODEL6-SVM
   SVM=SVC(probability=True)

   #VOTINGCLASSIFIER
   EVC1=VotingClassifier(estimators=[('LR',LR),('DTR',DTR),('KNN',KNN),('RFC',RFC),('SVM',SVM)],voting='hard')
   EVC1.fit(feature_train1,target_train1)
    #PREDICTION
   target_predevc1=EVC1.predict(feature_test1)
   name=st.text_input('Patient name:')
   col1,col2,col3,col4,col5=st.columns(5)
    
    
    

   with col1:
    #    st.header('MDVP:Fo(HZ)')
       A=st.number_input(label='Serum Creatinine(sc)')
   with col2:
       B=st.number_input(label='Blood Urea(bu)')
   with col3:
       C=st.number_input(label='Hemoglobin')
   with col4:
       D=st.number_input(label='Albumin(al)')
   with col5:
       
       E=st.number_input(label='BLoodPressure(bp)')
       
       
  
       
   input_data=np.array([[A,B,C,D,E]])
       
   if st.button('Predict'):
     predict1=EVC1.predict(input_data)
     
     if predict1=='ckd':
        st.error(f"Patient {name} has Kidney disease")
     else:
        st.success(f"Patient {name} does not have Kidney disease")

if selected=='Liver Prediction':
   col1,col2=st.columns(2)
   with col1:
    st.title(':red[LIVER DISEASE PREDICTION]')
   with col2:
    st.image(r'https://th.bing.com/th/id/OIP.VmIE-LIbfDAO0jxNMY_JxwHaEl?w=260&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7')
   df2=pd.read_csv(r"C:\Users\firea\Downloads\liver.csv")
   feature=df2[['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Aspartate_Aminotransferase','Albumin','Albumin_and_Globulin_Ratio']]
   target=df2['Dataset']

   #HANDLING OUTLIERS
   cools=feature.columns
   handledfeature=handle_outliers_iqr(feature,cools)

   #HANDLING NULL VALUES
   handledfeature['Albumin_and_Globulin_Ratio'].fillna(handledfeature['Albumin_and_Globulin_Ratio'].mean(),inplace=True)
   
   #BALANCING THE TARGET DATA
   feature_sampled2,target_y2=ros.fit_resample(handledfeature,target)

   #FEATURESCALING
   feature_x2=scaler2.fit_transform(feature_sampled2)
#PREPROCESSING COMPLETED

   feature_train2,feature_test2,target_train2,target_test2=train_test_split(feature_x2,target_y2)
   #ALGORITHM SELECTION
   #MODEL1-LOGISTIC REGRESSION
   LR=LogisticRegression(C=0.4,max_iter=1000,solver='liblinear')
  

    #MODEL2-DECISIONTREECLASSIFIER
   DTR=DecisionTreeClassifier()

    #MODEL4-KNN
   KNN=KNeighborsClassifier()
   

    #MODEL5-RANDOMFORESTCLASSIFIER
   RFC=RandomForestClassifier(criterion='entropy',random_state=14)
   
    #MODEL6-SVM
   SVM=SVC(probability=True)

   #VOTINGCLASSIFIER
   EVC2=VotingClassifier(estimators=[('LR',LR),('DTR',DTR),('KNN',KNN),('RFC',RFC),('SVM',SVM)],voting='hard')
   EVC2.fit(feature_train2,target_train2)
    #PREDICTION
   target_predevc2=EVC2.predict(feature_test2)
   name=st.text_input('Patient name:')
   col1,col2,col3,col4,col5=st.columns(5)
    
    
    

   with col1:
    #    st.header('MDVP:Fo(HZ)')
       A=st.number_input(label='Bilirubin')
       F=st.number_input(label='Albumin & Globulin')
   with col2:
       B=st.number_input(label='Dir_Bilirubin')
   with col3:
       C=st.number_input(label='Alkaline_Phos')
   with col4:
       D=st.number_input(label='Aminotransferate')
   with col5:
       
       E=st.number_input(label='Albumin')
       
       
  
       
   input_data=np.array([[A,B,C,D,E,F]])
       
   if st.button('Predict'):
     predict2=EVC2.predict(input_data)
     
     if predict2=='1':
        st.error(f"Patient {name} has Liver disease")
     else:
        st.success(f"Patient {name} does not have Liver disease")






