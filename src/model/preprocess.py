import advertools as adv
from string import punctuation,digits
from trnlp import TrnlpWord,SpellingCorrector
import re
from nltk import word_tokenize
import pandas as pd



getbase_word = TrnlpWord()
corrector = SpellingCorrector()
duraklama_kelimeleri = sorted(adv.stopwords['turkish'])[:]

def turkish_char(text):
    translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    result = text.translate(translationTable)
    return result.lower()


def change_correct(text):
    cwords=list()
    words = text.split()
    for i in words:
      corrector.settext(i)
      correct_word=corrector.correction(deasciifier=True)
      cwords.append(correct_word[0][0])
    cw = " ".join([cword for cword in cwords])
    return cw

def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation)
    text = text.translate(translator)
    translator = str.maketrans('', '', digits)
    return text.translate(translator)

def stop_word(text):
    text = word_tokenize(text, language='turkish')
    text = [word for word in text if not word in duraklama_kelimeleri]
    text = " ".join(text)
    return text


def get_stem(text):
    stems=list()
    words = text.split()
    for i in words:
      temp=i
      getbase_word.setword(i)
      stem = getbase_word.get_stem
      if stem =="":
        stem = temp
      stems.append(stem)
    sentence = " ".join([stem for stem in stems])
    return sentence

def greater_n(text):
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    word = shortword.sub('', text) 
    word = word.lower()
    return word

def cok_az_gecen(text):
    cok_gecenler = pd.Series(" ".join(text).split()).value_counts()[:1]
    az_gecenler = pd.Series(" ".join(text).split()).value_counts()
    az_gecenler=az_gecenler.loc[az_gecenler.values<=1]
    silinecek = pd.concat([cok_gecenler,az_gecenler])
    return  " ".join([word for word in text.split() if word not in silinecek])