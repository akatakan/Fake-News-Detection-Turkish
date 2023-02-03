import advertools as adv
from string import punctuation,digits
import pandas as pd 
import nltk
import re
from trnlp import TrnlpWord,SpellingCorrector
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.utils import shuffle

getbase_word = TrnlpWord()
corrector = SpellingCorrector()

duraklama_kelimeleri = sorted(adv.stopwords['turkish'])[:]

df = pd.read_csv(r"C:\Users\Atakan\Desktop\Fake-News-Detection-Turkish-main\news_dataset.csv",encoding="utf8")
df=df.drop(df.iloc[:,0:1],axis=1)
df = shuffle(df)
df = df.reset_index(drop=True)
df[df.duplicated(subset='Body')].count()
df.drop_duplicates(subset='Body', inplace=True)

def turkish_char(text):
    translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
    result = text.translate(translationTable)
    return result.lower()


def change_correct(text):
    cwords=list()
    words = text.split()
    for i in words:
      corrector.settext(i)
      correct_word=corrector.correction(all=True)
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


cok_gecenler = pd.Series(" ".join(df["Body"]).split()).value_counts()[:1]
az_gecenler = pd.Series(" ".join(df["Body"]).split()).value_counts()
az_gecenler=az_gecenler.loc[az_gecenler.values<=1]
silinecek = pd.concat([cok_gecenler,az_gecenler])

def cok_az_gecen(text):
  return  " ".join([word for word in text.split() if word not in silinecek])

textList = df.Body.apply(remove_punctuation)
textList = textList.apply(change_correct)
textList = textList.apply(get_stem)
textList = textList.apply(stop_word)
textList = textList.apply(turkish_char)
textList = textList.apply(greater_n)
textList = textList.apply(cok_az_gecen)



textList = list(textList)

df['clean_data'] = textList


df = df.drop(df.iloc[:,:1],axis=1)

df.to_csv("clean2.csv")



