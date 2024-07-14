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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
        return ' ' .join(hasil)
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

# Tambahkan fitur unggah file
st.subheader('Unggah File Ulasan')
uploaded_file = st.file_uploader("Unggah file ulasan dalam format CSV", type="csv")

# Fungsi untuk memproses file yang diunggah
def process_uploaded_file(file):
    df = pd.read_csv(file)
    if 'content' in df.columns:
        return df['content'].tolist()
    else:
        st.error("File tidak memiliki kolom 'content'. Pastikan file memiliki kolom 'content' yang berisi ulasan.")
        return []

# Tambahkan tombol "Tampilkan Hasil Sentimen" yang selalu ada
if st.button('Tampilkan Hasil Sentimen'):
    reviews_list = []
    if user_input:
        reviews_list = user_input.split('\n')
    
    if uploaded_file:
        reviews_list.extend(process_uploaded_file(uploaded_file))
    
    if reviews_list:
        scores = []
        for review in reviews_list:
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
        for i, count in enumerate(sentiment_counts.values):
            ax.text(i, count, str(count), ha='center', va='bottom', fontsize=12)  # Menambahkan total di atas setiap bar
        ax.set_xlabel('Sentimen', fontsize=12)
        ax.set_ylabel('Jumlah', fontsize=12)
        ax.set_title('Distribusi Sentimen', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)  # Sesuaikan parameter tick
        st.pyplot(fig)
    else:
        st.warning('Silakan masukkan ulasan atau unggah file untuk menganalisis sentimen.')

# Ambil ulasan dari Google Play Store
st.subheader('Ambil Ulasan dari Google Play Store')
number_of_reviews = st.selectbox('Pilih jumlah ulasan yang ingin diambil', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500], index=2)
if st.button('Ambil Ulasan'):
    with st.spinner('Mengambil ulasan...'):
        reviews_data = []
        for score in range(1, 6):
            rvs, _ = reviews(
                'id.co.aviana.m_pulsa',
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=number_of_reviews // 10,
                filter_score_with=score
            )
            reviews_data.extend(rvs)

        reviews_df = pd.DataFrame(reviews_data)
        reviews_df['content'] = reviews_df['content'].apply(clean_text)
        st.write(reviews_df.head())

        reviews_df['preprocessed_content'] = reviews_df['content'].apply(lambda x: preprocess_text(x, kamus_normalisasi_tidak, stopwords_indonesian, stopwords_exceptions, stemmer))
        st.write(reviews_df.head())

        scores = []
        for review in reviews_df['preprocessed_content']:
            score, polarity, affected_words = sentiment_lexicon(review, lexicon_positive, lexicon_negative)
            affected_words_str = ', '.join([f"{word} ({sentiment}: {value})" for word, value, sentiment in affected_words])
            scores.append((score, polarity, affected_words_str))

        reviews_df['score'], reviews_df['sentiment'], reviews_df['affected_words'] = zip(*scores)
        reviews_df['lexicon_sentiment'] = reviews_df['score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

        st.write(reviews_df[['content', 'lexicon_sentiment', 'affected_words']])

        sentiment_counts = reviews_df['lexicon_sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values)
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Sentimen')
        st.pyplot(fig)
