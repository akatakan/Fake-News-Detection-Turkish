import advertools as adv
from string import punctuation, digits
from trnlp import TrnlpWord
import re
import pandas as pd

getbase_word = TrnlpWord()
duraklama_kelimeleri = set(adv.stopwords['turkish'])


def turkish_char(text):
    translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    return text.translate(translationTable).lower()


def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation + digits)
    return text.translate(translator)


def get_stem(text):
    stems = []
    for word in text.split():
        getbase_word.setword(word)
        stem = getbase_word.get_stem
        stems.append(stem if stem else word)
    return " ".join(stems)


def stop_word(text):
    return " ".join(word for word in text.split() if word not in duraklama_kelimeleri)


def greater_n(text):
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    return shortword.sub('', text)


def cok_az_gecen(text):
    cok_gecenler = pd.Series(" ".join(text).split()).value_counts()[:1]
    az_gecenler = pd.Series(" ".join(text).split()).value_counts()
    az_gecenler = az_gecenler.loc[az_gecenler.values <= 1]
    silinecek = pd.concat([cok_gecenler, az_gecenler])
    return " ".join(word for word in text.split() if word not in silinecek)
