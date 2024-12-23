import streamlit as st
import numpy as np
import pandas as pd
import pickle 
from tensorflow.keras.models import load_model 
#import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
# load model
model=load_model('models/next_word_prediction.h5',compile=False)

# load tokenizer

with open ('models/tokenizer.pkl','rb') as f:
    tokenizer=pickle.load(f)


# function to predict the next word 
def predict_next_word(model,tokenizer,text,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]

  if len(token_list) >= max_sequence_len:
    token_list = token_list[max_sequence_len-1:]
  token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  prediction = model.predict(token_list,verbose=0)
  predicted_word_index = np.argmax(prediction,axis=1)
  # predicted_word = tokenizer.index_word[predicted_word_index]
  # return predicted_word
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None



# streamlit apps

st.title ("Next word prediction apps")
input_text=st.text_input("Enter the word!")
if st.button("Predict Next word"):
  max_sequence_len=model.input_shape[1]+1
  next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
  st.write(f'Next word id:{next_word}')