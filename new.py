import pandas as pd
import streamlit as st
import mysql.connector.connect


mydb=mysql.connector.connect(
    host='localhost',
    user='root',
    password='Jsa.5378724253@'
)
arun=mydb.cursor()
st.write('connected')