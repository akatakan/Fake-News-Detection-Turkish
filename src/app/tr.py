import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,SpatialDropout1D
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
nltk.download('words')


words = set(nltk.corpus.words.words("turkish"))
print(words)