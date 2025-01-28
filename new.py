import pandas as pd
import streamlit as st
import boto3
from io import StringIO

# S3 file details
st.title('S3 Data Extract')
bucket_name = 'multiple-disease-s3'
path1 = 'Kidney.csv'
def get_s3_file(bucket_name, path1):
    try:
        s3 = boto3.client('s3')  # No explicit credentials needed if IAM role is used
        response = s3.get_object(Bucket=bucket_name, Key=path1)
        data = response['Body'].read().decode('utf-8')
        return data
    except Exception as e:
        st.error(f"Error accessing S3 file: {e}")
        return None

file1 = get_s3_file(bucket_name, path1)
if file1:
 
  df1 = pd.read_csv(StringIO(file1))
  st.dataframe(df1)

path2 = 'liver.csv'
def get_s3_file(bucket_name, path2):
    try:
        s3 = boto3.client('s3')  # No explicit credentials needed if IAM role is used
        response = s3.get_object(Bucket=bucket_name, Key=path2)
        data = response['Body'].read().decode('utf-8')
        return data
    except Exception as e:
        st.error(f"Error accessing S3 file: {e}")
        return None


file2 = get_s3_file(bucket_name, path2)
if file2:
 
  df2 = pd.read_csv(StringIO(file2))
  st.dataframe(df2)



