import streamlit as st
import pandas as pd
import numpy as np

st.title('Machine Learning App')

st.info('This is app builts a machine learning model')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species',axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw


with st.expander('Data Visualization'):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# Data Preparation

with st.sidebar:
  st.header('Input Features')
  island= st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
  sex = st.selectbox('Gender',('male','female'))


  # Create dataframe for input features
  data = {'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex
        }

  df_input = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([df_input,X], axis=0)

# Encode X categorical features
encoded = ['island', 'sex']
df_penguins=pd.get_dummies(input_penguins,prefix=encoded)
input_row = df_penguins[:1] #show first row only

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2            
                 }

def target_encode(val):
  return target_mapper[val]



with st.expander('Input Features'):
  st.write('**Input penguin**')
  df_input
  st.write('**Combined Penguins Data**')
  input_penguins
  st.write('**Encoded Input Penguin**')
  input_row

