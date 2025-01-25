import pandas as pd
import streamlit as st
import boto3
from io import StringIO

# S3 file details
st.title('S3 Data Extract')
bucket_name = 'multiple-disease-s3'
path = 'parkinsons.csv'
def get_s3_file(bucket_name, path):
    try:
        s3 = boto3.client('s3')  # No explicit credentials needed if IAM role is used
        response = s3.get_object(Bucket=bucket_name, Key=path)
        data = response['Body'].read().decode('utf-8')
        return data
    except Exception as e:
        st.error(f"Error accessing S3 file: {e}")
        return None
with st.sidebar:
   select=st.selectbox('select',['nav','arun'])
if select=='nav':
 file = get_s3_file(bucket_name, path)
 if file:
 
  df = pd.read_csv(StringIO(file))
  st.dataframe(df)

# Streamlit App



