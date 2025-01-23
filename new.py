import pandas as pd
import streamlit as st
import mysql.connector


mydb=mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jsa.5378724253@'
)
arun=mydb.cursor()
st.write('connected')

path='s3://multiple-disease-s3/Kidney.csv'

df=pd.read_csv(path)
st.dataframe(df)
