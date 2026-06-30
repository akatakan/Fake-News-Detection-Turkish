# Türkçe Sahte Haber Tespiti

Türkçe haber metinlerini **sahte** veya **gerçek** olarak sınıflandıran bir LSTM modelidir. Flask web arayüzü üzerinden kullanılabilir.

## Model

- **Mimari**: LSTM (64 birim) + SpatialDropout + Dense(sigmoid)
- **Eğitim verisi**: ~4500 haber (zaytung.com + hurriyet.com.tr)
- **Test doğruluğu**: %97
- **Ön işleme**: Noktalama temizleme → yazım düzeltme → kök bulma → stop-word → Türkçe karakter normalizasyonu

## Kurulum

```bash
git clone https://github.com/akatakan/Fake-News-Detection-Turkish
cd Fake-News-Detection-Turkish
uv sync
```

## Kullanım

### Web arayüzü

```bash
uv run flask --app src.app run
```

`http://127.0.0.1:5000` adresini aç, haberi yapıştır, **Analiz Et** butonuna bas.

### Modeli yeniden eğit

```bash
uv run python src/app/LSTM.py
```

Eğitim tamamlandığında `src/optimized_model.h5`, `src/tokenizer.pkl` ve `src/max_tokens.txt` dosyaları oluşturulur.

### Veri topla

```bash
python src/app/fake.py       # zaytung.com
python src/app/real.py       # hurriyet.com.tr
python src/app/preprocessing.py  # temizle ve birleştir
```

## Proje Yapısı

```
├── src/
│   ├── app.py                  # Flask uygulaması
│   ├── app/
│   │   ├── LSTM.py             # Model eğitimi
│   │   ├── preprocessing.py    # Veri temizleme
│   │   ├── fake.py             # Sahte haber scraper
│   │   └── real.py             # Gerçek haber scraper
│   ├── model/
│   │   ├── prediction_model.py # Inference
│   │   └── preprocess.py       # Metin ön işleme
│   ├── templates/index.html
│   └── static/styles/style.css
├── clean2.csv                  # Temizlenmiş eğitim verisi
└── news_dataset.csv            # Ham veri
```

## Lisans

MIT
