import streamlit as st
import tensorflow as tf
from preprocess import masked_loss, masked_accuracy, predict

model_path = './model/model.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'masked_loss': masked_loss, 'masked_accuracy': masked_accuracy})

sentence_vectorizer = './vectorizer'
vectorizer = tf.keras.layers.TextVectorization(standardize = None)
vectorizer.load_assets(sentence_vectorizer)

st.title('Named Entity Recognition Prediction')
text = st.text_area('Input Text : ')
button = st.button('Predict')

if button:
    st.subheader('Predictions : ')
    results = predict(text, model, vectorizer)
    for x,y in zip(text.split(' '), results):
        if y != 'O':
            st.write(f'{x} --> {y}')