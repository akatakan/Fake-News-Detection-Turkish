import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import random
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

ROOT = Path(__file__).parent.parent.parent
SRC  = Path(__file__).parent.parent

df = pd.read_csv(ROOT / "clean2.csv", encoding="utf8")

fake  = df.loc[df["Label"] == 0]["clean_data"]
real  = df.loc[df["Label"] == 1]["clean_data"]
y_fake = df.loc[df["Label"] == 0]["Label"]
y_real = df.loc[df["Label"] == 1]["Label"]

fake_cutoff = int(len(fake) * 0.8)
real_cutoff = int(len(real) * 0.8)

X      = pd.concat([fake[:fake_cutoff], real[:real_cutoff]])
X_test = pd.concat([fake[fake_cutoff:], real[real_cutoff:]])
y_train_s = pd.concat([y_fake[:fake_cutoff], y_real[:real_cutoff]])
y_test    = np.asarray(pd.concat([y_fake[fake_cutoff:], y_real[real_cutoff:]]))

# fake/real sırasını kır — validation split dengeli olsun
shuffle_idx = np.random.permutation(len(X))
X       = X.iloc[shuffle_idx].reset_index(drop=True)
y_train = np.asarray(y_train_s)[shuffle_idx]

X_egitim = [metin.split() for metin in X]

tokenizer_full = Tokenizer()
tokenizer_full.fit_on_texts(X_egitim)
vocab_size = len(tokenizer_full.word_index)

word_counts = sorted(tokenizer_full.word_counts.values(), reverse=True)
cumsum = np.cumsum(word_counts) / sum(word_counts)

print(f"\nToplam vocab boyutu: {vocab_size}")
print("Coverage analizi:")
for n in [1000, 2000, 3000, 5000, 8000]:
    if n <= vocab_size:
        print(f"  num_words={n:5d} → coverage={cumsum[n-1]*100:.1f}%")

threshold_95 = int(np.searchsorted(cumsum, 0.95)) + 1
num_words = int(np.ceil(threshold_95 / 1000) * 1000)
num_words = min(num_words, vocab_size)
print(f"\n%95 coverage için gereken: {threshold_95} → num_words={num_words} olarak ayarlandı")

raw_dim = num_words ** 0.25
embedding_dim = next(p for p in [8, 16, 32, 64, 128] if p >= raw_dim)
print(f"embedding_dim: {num_words}^0.25 ≈ {raw_dim:.1f} → {embedding_dim}")

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_egitim)

X_train_tok = tokenizer.texts_to_sequences(X_egitim)
X_test_tok  = tokenizer.texts_to_sequences(X_test)

lengths    = np.array([len(t) for t in X_train_tok + X_test_tok])
max_tokens = int(np.mean(lengths) + 2 * np.std(lengths))
print(f"max_tokens: {max_tokens}\n")

X_train_pad = pad_sequences(X_train_tok, maxlen=max_tokens)
X_test_pad  = pad_sequences(X_test_tok,  maxlen=max_tokens)


def build_original(num_words, embedding_dim, max_tokens):
    m = Sequential(name="original_lstm")
    m.add(Embedding(input_dim=num_words, output_dim=embedding_dim,
                    input_length=max_tokens, name="embedding"))
    m.add(SpatialDropout1D(0.2))
    m.add(LSTM(32, return_sequences=True))
    m.add(LSTM(16, return_sequences=True))
    m.add(LSTM(8))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(3e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_optimized(num_words, embedding_dim, max_tokens):
    m = Sequential(name="optimized_lstm")
    m.add(Embedding(input_dim=num_words, output_dim=embedding_dim,
                    input_length=max_tokens, name="embedding"))
    m.add(SpatialDropout1D(0.3))
    m.add(LSTM(64))
    m.add(Dropout(0.3))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(3e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return m


BATCH_SIZE = 32

early_stop_original = EarlyStopping(
    monitor="val_accuracy", patience=2,
    restore_best_weights=True, verbose=1
)

early_stop_optimized = EarlyStopping(
    monitor="val_accuracy", patience=3,
    restore_best_weights=True, verbose=1
)

print("=" * 50)
print("Model 1: Original LSTM + SpatialDropout (max 10 epoch, patience=2)")
print("=" * 50)
model_original = build_original(num_words, embedding_dim, max_tokens)
model_original.summary()
history_original = model_original.fit(
    X_train_pad, y_train,
    epochs=10, batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop_original]
)

print("\n" + "=" * 50)
print("Model 2: Optimized LSTM (max 15 epoch, patience=3)")
print("=" * 50)
model_optimized = build_optimized(num_words, embedding_dim, max_tokens)
model_optimized.summary()
history_optimized = model_optimized.fit(
    X_train_pad, y_train,
    epochs=15, batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop_optimized]
)

loss_o, acc_o = model_original.evaluate(X_test_pad, y_test, verbose=0)
loss_n, acc_n = model_optimized.evaluate(X_test_pad, y_test, verbose=0)

print("\n" + "=" * 50)
print("TEST SONUÇLARI")
print("=" * 50)
print(f"  Original  → loss: {loss_o:.4f}  accuracy: {acc_o*100:.2f}%")
print(f"  Optimized → loss: {loss_n:.4f}  accuracy: {acc_n*100:.2f}%")

val_acc_o = history_original.history["val_accuracy"]
val_acc_n = history_optimized.history["val_accuracy"]
print(f"\n  Original  val_accuracy per epoch:  {[round(v*100,1) for v in val_acc_o]}")
print(f"  Optimized val_accuracy per epoch:  {[round(v*100,1) for v in val_acc_n]}")

model_original.save(SRC / "haber_tespit_model.h5")
model_optimized.save(SRC / "optimized_model.h5")

with open(SRC / "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open(SRC / "max_tokens.txt", "w") as f:
    f.write(str(max_tokens))

print(f"\nModeller kaydedildi → {SRC}")
