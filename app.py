%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
max_seq_len = int(open("seq_length.txt").read())

def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return "No word predicted"

# Streamlit UI
st.title("🧠 Next Word Predictor")
st.write("Enter a phrase and get the next predicted word using LSTM!")

user_input = st.text_input("Enter your text here:")
if user_input:
    next_word = predict_next_word(user_input)
    st.success(f"Next word: {next_word}")
