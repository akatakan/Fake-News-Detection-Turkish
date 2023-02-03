from tensorflow.keras.models import load_model
from model.preprocess import remove_punctuation,get_stem,change_correct,stop_word,turkish_char,greater_n,cok_az_gecen
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore")



news_model = load_model(r"C:\Users\Atakan\Desktop\Fake-News-Detection-Turkish-main\src\haber_tespit_model.h5")

def preprocess(text):
    text = remove_punctuation(text)
    text = change_correct(text)
    text = get_stem(text)
    text = stop_word(text)
    text = turkish_char(text)
    text = greater_n(text)
    text = cok_az_gecen(text)
    return text

def tokenizing_and_pad(text):
    splitted_text = [text.split()]
    tokenizer = Tokenizer()
    text_tok = tokenizer.texts_to_sequences(splitted_text)
    text_pad = pad_sequences(text_tok, maxlen=295)
    return text_pad

def predict(text):
    pre_text = preprocess(text)
    pad_text = tokenizing_and_pad(pre_text)
    prob = news_model.predict(pad_text)
    pred=prob[0].round()
    print(pred,"-",prob[0])
    if pred ==1:
        return True,round(prob[0][0]*100)
    else:
        return False,round(100-(prob[0][0]*100))

