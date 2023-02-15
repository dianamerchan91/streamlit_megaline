import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn

data = pd.read_csv('clean_df.csv')
modelo = pickle.load(open('modelo.pickle', 'rb'))

# Título

st.title('¿Cuál es tu plan de telefonía móvil ideal?')

# Exploración inicial

st.header('Exploración Inicial')
st.subheader('Los primeros datos')
st.dataframe(data.head())

st.subheader('Un descriptivo')
st.dataframe(data.describe())

# Visualización

st.header('Visualización de datos')

fig, ax = plt.sbuplots(1, 4), sharey=True, figsize=(16,4))
ax[0].set_ylabel('%')

for idx, col in enumerate(['calls', 'minutes', 'messages', 'mb_used']):
  data[col].plot(kind='hist', ax=ax[idx], title=col)
plt.show()
  
st.pyplot(fig)
# Selección de datos

st.header('¿Cuántos mensajes, llamadas y datos consumes al mes?')
col1, col2  =  st.columns(2)

with col1:
  llamadas = st.slider('Número de llamadas al mes', 0, 244)
  minutos = st.slider('Minutos utilizados al mes', 0, 1632)
  
with col2:
  mensajes = st.slider('Mensajes al mes', 0, 224)
  mb_used = st.slider('MB usados al mes, 0, 49745)
                      
# Predicción
if st.button('Predecir'):
  pred = modelo.predict(np.array([[llamadas, minutos, mensajes, mb_used]]))
  st.text(f'Tu plan de telefonía movil ideal es: {pred})
else:
  st.text('Seleccione entre las opciones e imprima predecir')
                      
                      
