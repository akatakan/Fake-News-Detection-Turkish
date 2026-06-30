from tensorflow.keras.models import load_model
from model.preprocess import remove_punctuation, get_stem, change_correct, stop_word, turkish_char, greater_n
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import numpy as np
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

SRC = Path(__file__).parent.parent

news_model = load_model(SRC / "optimized_model.h5")

with open(SRC / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(SRC / "max_tokens.txt") as f:
    max_tokens = int(f.read().strip())


def preprocess(text):
    text = remove_punctuation(text)
    text = change_correct(text)
    text = get_stem(text)
    text = stop_word(text)
    text = turkish_char(text)
    text = greater_n(text)
    return text


def tokenizing_and_pad(text):
    splitted_text = [text.split()]
    text_tok = tokenizer.texts_to_sequences(splitted_text)
    text_pad = pad_sequences(text_tok, maxlen=max_tokens)
    return text_pad


def predict(text):
    pre_text = preprocess(text)
    pad_text = tokenizing_and_pad(pre_text)
    prob = news_model.predict(pad_text)
    pred = prob[0].round()
    if pred == 1:
        return True, round(prob[0][0] * 100)
    else:
        return False, round(100 - (prob[0][0] * 100))
