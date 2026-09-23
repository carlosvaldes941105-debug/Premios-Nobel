import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.write(''' # What is your motivation for winning a Nobel Prize? ''')
st.image("istockphoto-1492930188-612x612.jpg", caption="Nobel Prize winner's medal")

st.header('Texto')
def user_input_features():
  # Entrada
  texto = st.text_input("Write a statement explaining why you deserve to be awarded the Nobel Prize.")

  user_input_data = {'Text': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
user_text = df['Text'].iloc[0]

nobel =  pd.read_csv('premios_nobel_ok.csv', encoding='latin-1')
X = nobel.Mot_limp
y = nobel.num_label

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

nb = MultinomialNB()
nb.fit(X_dtm, y)

#sin_duplicados['Lable_num']=sin_duplicados.Category.map({"chemistry":1,"economics":2,"literature":3,"medicine":4,"peace":5,"physics":6})

st.subheader('The prediction of the prize based on your motivation is:')
if not user_text or not user_text.strip():
  st.write('No forecast')
else:
  words = user_text.split()
  if len(words) < 4:
    st. write('No forecast')
  else:
    df_dtm = vect.transform([user_text])
    prediction = nb.predict(df_dtm)[0]
    confidence = nb.predict_proba(df_dtm)[0].max()
    if confidence < 0.5:
      st.write('No forecast')
    elif prediction == 1:
      st.write('Chemistry')
    elif prediction == 2:
      st.write('Economics')
    elif prediction == 3:
      st.write('Literature')
    elif prediction == 4:
      st.write('Medicine')
    elif prediction == 5:
      st.write('Peace')
    elif prediction == 6:
      st.write('Physics')
    else:
      st.write('No forecast')
