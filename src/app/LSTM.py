import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,SpatialDropout1D
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv(r"C:/Users/Atakan/Desktop/Fake-News-Detection-Turkish-main/clean2.csv",encoding="utf8")

fake = df.loc[df["Label"]==0]["clean_data"]
real = df.loc[df["Label"]==1]["clean_data"]
y_fake = df.loc[df["Label"]==0]["Label"]
y_real = df.loc[df["Label"]==1]["Label"]

fake_cutoff = int(len(fake)*0.8)
real_cutoff = int(len(real)*0.8)

fake_train,fake_test = fake[:fake_cutoff],fake[fake_cutoff:]
real_train,real_test = real[:real_cutoff],real[real_cutoff:]
y_train=pd.concat([y_fake[:fake_cutoff],y_real[:real_cutoff]])
y_test=pd.concat([y_fake[fake_cutoff:],y_real[real_cutoff:]])


X = pd.concat([fake_train,real_train])
X_test = pd.concat([fake_test, real_test])
X_egitim = [metin.split() for metin in X]

num_words=1000
tokenizer = Tokenizer(num_words=num_words) 
tokenizer.fit_on_texts(X_egitim)
X_train_token = tokenizer.texts_to_sequences(X_egitim)
X_test_token = tokenizer.texts_to_sequences(X_test)
kelime_index = tokenizer.word_index


# Tüm dataset'indeki vocab (sözlük) kelime sayısı.
kelime_sayi = len(kelime_index) + 1

print('Sözlük boyutu: ', kelime_sayi)

x_train_pad = pad_sequences(X_train_token)
x_test_pad = pad_sequences(X_test_token)


embedding_size = 50
num_tokens = [len(tokens) for tokens in X_train_token + X_test_token]
num_tokens = np.array(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

x_test_pad = pad_sequences(X_test_token, maxlen=max_tokens)
X_train_pad = pad_sequences(X_train_token, maxlen=max_tokens)

x_train_pad = np.asarray(x_train_pad)
y_train = np.asarray(y_train)

model = Sequential()
model.add(Embedding(
    input_dim = num_words,
    output_dim = embedding_size,
    input_length = max_tokens,
    name = 'embedding_layer'
))
model.add(LSTM(units=32, return_sequences=True))
model.add(LSTM(units=16, return_sequences=True))
model.add(LSTM(units=8))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

result = model.fit(
        X_train_pad,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2
)


news="instagramda kisa sure bin takipci ulasan burcu oytun basari sirri paylas guzel yaklasik katildik instagram camia ant fenomen hali gelen burcu oytun kisa zaman gelen muthis basari art gercek kamuoyu paylas instagram koyduk fotograf sanatsal deger onem olmadik samimiyet ifade ede oytun tas bay bildik guzel resmen durum aciklik getir basarili guzel takipci sayi basin toplanti adet art sosyal medya gelen platform instagram rekor elde ettik bin takipci ant ilgi odak hali gelen burcu oytun duzenledik basin toplanti seven kamuoyu seslen konusma degerli basin mensup arkadas hooop evet yuzum etek kafa kaldirabil laf edecegiz seklî gazeteci takilarak baslayan oytun hayli neseli yemek foto falan diyerek âdeta doken guzel genc populerlik elde ettik ornek acikla buyrun sabahtan kare ayni yumurta ayni kahvalti mekân kader selvi isimli kullanici paylas like like pardon aha say nazar deg olay asagi yukari gelis dusun bulduk yanit hayvan guzel ahah hay masallah sosyal medya begenilen yuz instagramda fenomen isteyen kullanici seslenme ihmal tabir loser lara fotografcilik kurs gitmek goruntu ayarlamak goster filtre uygulamak secenek oneren burcu oytun deney bil sor instagram kullanici estetik ameliyat secenek dur tip diyerek onemli tavsiye bulun son gazeteci toplu selfie ceken yukleyen deneyimli instagram fenomen basarili filtre kullanim goster oytun bak arkadas buyrun filtre sec siyahbeyaz yap dile gorduk guzel yanim iki arkadas afedersiniz kurbaga isik ayni isik ayni takipci son mesaj verdi ara uzeri arquedesignin hediye nisantasindaki magaza mutlaka ziyaret edin harika model metin firat kuafor yap muhtesem kuafor tesvikiyede kes gidi ayakkabi sponsor"
tokens = tokenizer.texts_to_sequences(news)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
y_pred= model.predict(x_test_pad)

cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test)
incorrect_preds = np.where(cls_pred != cls_true)
incorrect_preds = incorrect_preds[0]
print(len(incorrect_preds))
idx = incorrect_preds[0]
first_incorrect_pred = X_test.iloc[idx]
print(first_incorrect_pred)
model.save("haber_tespit_model.h5")
loss, accuracy = model.evaluate(x_test_pad, y_test)
print('Accuracy: %f' % (accuracy*100))

