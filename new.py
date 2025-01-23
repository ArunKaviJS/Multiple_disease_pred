import pandas as pd
import streamlit as st

path='s3://multiple-disease-s3/Kidney.csv'

df=pd.read_csv(path)
st.dataframe(df)
