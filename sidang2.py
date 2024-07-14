import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import pickle
import os
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm
from google_play_scraper import Sort, reviews

# Set the NLTK data path
nltk.data.path.append('./nltk_data')

# Fungsi bantu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return ' '.join(dict.fromkeys(text.split()))

def normalisasi(teks, kamus):
    return ' '.join([kamus.get(k, k) for k in teks.split()]) if pd.notna(teks) else teks

def hapus_stopwords(teks, stopwords_indonesian, stopwords_exceptions):
    return ' '.join([word for word in str(teks).split() if word.lower() not in stopwords_indonesian or word.lower() in stopwords_exceptions])

def clean_and_combine(text):
    return ' '.join(text)

def read_lexicon(file_path):
    return pd.read_excel(file_path).set_index('kata')['skor'].to_dict()

def sentiment_lexicon(text, lexicon_positive, lexicon_negative):
    score = 0
    affected_words = []
    for word in text.split():
        if (lexicon_positive is not None and word in lexicon_positive):
            score += lexicon_positive[word]
            affected_words.append((word, lexicon_positive[word], 'positive'))
        elif (lexicon_negative is not None and word in lexicon_negative):
            score += lexicon_negative[word]
            affected_words.append((word, lexicon_negative[word], 'negative'))
    polarity = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    return score, polarity, affected_words

def sentiment_mapping(polarity):
    return {'positive': 1, 'negative': -1, 'neutral': 0}[polarity]

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def train_evaluate_svm(X, y, kernel, C, cv):
    svm = SVC(kernel=kernel, C=C, random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(svm, X, y, cv=skf, scoring='accuracy')
    svm.fit(X, y)
    y_pred = cross_val_score(svm, X, y, cv=skf, scoring='accuracy')
    cm = confusion_matrix(y, svm.predict(X))
    return scores, cm

def evaluate_svm_kfold(X, y, k=10, kernel='linear', C=1.0, gamma='scale'):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=-1, return_train_score=False)
    return scores, kf.split(X, y), model

def process_fold(X, y, train_index, test_index, model):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Fungsi normalisasi untuk frasa "tidak"
def cari_dan_normalisasi_tidak(teks, kamus):
    if isinstance(teks, str):
        kata_kata = teks.split()
        hasil = []
        i = 0
        while i < len(kata_kata):
            if kata_kata[i].lower() == 'tidak' and i + 1 < len(kata_kata):
                frasa = f"{kata_kata[i]} {kata_kata[i+1]}"
                hasil.append(kamus.get(frasa, frasa))
                i += 2  # Lompat ke kata setelah frasa "tidak"
            else:
                hasil.append(kata_kata[i])
                i += 1
        return ' '.join(hasil)
    return teks

# Fungsi untuk memisahkan imbuhan "-nya"
def pisahkan_imbuhan_nya(teks):
    if isinstance(teks, str):
        kata_kata = teks.split()
        hasil = []
        for kata in kata_kata:
            if kata.endswith("nya"):
                kata_dasar = kata[:-3]
                imbuhan = "nya"
                hasil.append(kata_dasar)
                hasil.append(imbuhan)
            else:
                hasil.append(kata)
        return ' '.join(hasil)
    return teks

# Konfigurasi aplikasi Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Aplikasi MPStore",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Terapkan CSS untuk styling
st.markdown(
    """
    <style>
    .main {
        background-color: #e6f7ff;
        color: #004080;
    }
    .sidebar .sidebar-content {
        background-color: #f0f5ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tambahkan logo dan judul
logo_path = "imam.png"
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" width="50" height="50" style="margin-right: 10px;">
        <h1 style="color: #004080;">Analisis Sentimen Ulasan Aplikasi MPStore</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Muat model jika tersedia
model_path = 'sentiment_model.pkl'
if os.path.exists(model_path):
    model = load_model(model_path)
    lexicon_positive = model['lexicon_positive']
    lexicon_negative = model['lexicon_negative']
    st.success('Model berhasil dimuat dari sentiment_model.pkl')
else:
    lexicon_positive = read_lexicon('kamus/positive_lexicon.xlsx')
    lexicon_negative = read_lexicon('kamus/negative_lexicon.xlsx')
    st.error('File sentiment_model.pkl tidak ditemukan. Pastikan untuk menjalankan dan menyimpan model terlebih dahulu.')

# Definisikan kamus normalisasi "tidak"
kamus_normalisasi_tidak = pd.read_excel('kamus/kamus_tidak.xlsx').set_index('frasa')['normalisasi'].to_dict()

# Load stopwords
stopwords_df = pd.read_excel('kamus/stopword_dictionary.xlsx')
stopwords_indonesian = stopwords_df['stopword'].tolist()
stopwords_exceptions = []

# Buat objek stemmer di sini
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text, kamus_normalisasi_tidak, stopwords_indonesian, stopwords_exceptions, stemmer):
    text = clean_text(text)
    text = pisahkan_imbuhan_nya(text)
    text = cari_dan_normalisasi_tidak(text, kamus_normalisasi_tidak)
    text = hapus_stopwords(text, stopwords_indonesian, stopwords_exceptions)
    text = stemmer.stem(text)
    return text

# Input pengguna untuk ulasan baru
st.subheader('Analisis Ulasan Baru')
user_input = st.text_area('Masukkan ulasan Anda (satu per baris) id.co.aviana.m_pulsa', '')

# Tambahkan tombol "Tampilkan Hasil Sentimen" yang selalu ada
if st.button('Tampilkan Hasil Sentimen'):
    if user_input:
        scores = []
        for review in user_input.split('\n'):
            review_preprocessed = preprocess_text(review, kamus_normalisasi_tidak, stopwords_indonesian, stopwords_exceptions, stemmer)
            score, polarity, affected_words = sentiment_lexicon(review_preprocessed, lexicon_positive, lexicon_negative)
            affected_words_str = ', '.join([f"{word} ({sentiment}: {value})" for word, value, sentiment in affected_words])
            scores.append((review, score, polarity, affected_words_str))

        # Membuat DataFrame untuk menampilkan hasil dalam bentuk tabel
        result_df = pd.DataFrame(scores, columns=['content', 'score', 'sentiment', 'affected_words'])
        result_df['lexicon_sentiment'] = result_df['score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

        st.write(result_df[['content', 'lexicon_sentiment', 'affected_words']])

        # Visualisasikan distribusi sentimen
        st.subheader('Distribusi Sentimen')
        sentiment_counts = result_df['lexicon_sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  # Sesuaikan ukuran dan DPI gambar
        ax.bar(sentiment_counts.index, sentiment_counts.values, width=0.4)  # Sesuaikan lebar bar jika diperlukan
        ax.set_xlabel('Sentimen', fontsize=12)
        ax.set_ylabel('Jumlah', fontsize=12)
        ax.set_title('Distribusi Sentimen', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)  # Sesuaikan parameter tick
        st.pyplot(fig)
    else:
        st.warning('Silakan masukkan ulasan untuk menganalisis sentimen.')

# Sidebar untuk scraping ulasan dari Google Play Store
st.sidebar.subheader('Scraping Ulasan dari Google Play Store')
app_id_input = st.sidebar.text_input('Masukkan ID Aplikasi Google Play Store', '')
jumlah_ulasan_input = st.sidebar.number_input('Jumlah ulasan yang ingin di-scrape', min_value=1, max_value=1000, value=100, step=10)

if st.sidebar.button('Scrape Ulasan'):
    if app_id_input:
        # Scrap ulasan dari Google Play Store
        scraped_reviews, _ = reviews(app_id_input, lang='id', count=jumlah_ulasan_input, sort=Sort.MOST_RELEVANT)
        scraped_data = pd.DataFrame(scraped_reviews)

        # Membersihkan teks ulasan
        scraped_data['clean'] = scraped_data['content'].apply(lambda x: preprocess_text(x, kamus_normalisasi_tidak, stopwords_indonesian, stopwords_exceptions, stemmer))

        # Analisis sentimen
        scraped_data['sentiment_score'] = scraped_data['clean'].apply(lambda x: sentiment_lexicon(x, lexicon_positive, lexicon_negative)[0])
        scraped_data['sentiment_polarity'] = scraped_data['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

        # Konversi affected_words menjadi string
        scraped_data['affected_words'] = scraped_data['clean'].apply(lambda x: sentiment_lexicon(x, lexicon_positive, lexicon_negative)[2])
        scraped_data['affected_words'] = scraped_data['affected_words'].apply(lambda words: ', '.join([f"{word} ({sentiment}: {value})" for word, value, sentiment in words]))

        # Tampilkan hasil pada aplikasi
        st.sidebar.subheader('Hasil Sentimen Ulasan:')
        st.sidebar.write(scraped_data[['content', 'sentiment_polarity', 'affected_words']])

        # Visualisasikan distribusi sentimen
        st.sidebar.subheader('Distribusi Sentimen:')
        sentiment_counts = scraped_data['sentiment_polarity'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  # Sesuaikan ukuran dan DPI gambar
        ax.bar(sentiment_counts.index, sentiment_counts.values, width=0.4)  # Sesuaikan lebar bar jika diperlukan
        ax.set_xlabel('Sentimen', fontsize=12)
        ax.set_ylabel('Jumlah', fontsize=12)
        ax.set_title('Distribusi Sentimen', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)  # Sesuaikan parameter tick
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.warning('Silakan masukkan ID Aplikasi Google Play Store.')

# Footer
st.sidebar.markdown("---")
