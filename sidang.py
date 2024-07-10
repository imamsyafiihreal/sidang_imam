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

# Clone the GitHub repository
repo_url = 'https://github.com/imamsyafiihreal/sidang_imam'
repo_dir = 'sidang_imam'

if not os.path.exists(repo_dir):
    subprocess.run(['git', 'clone', repo_url])

# Tentukan lokasi direktori nltk_data lokal
nltk_data_path = os.path.join(repo_dir, 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Sekarang unduh data yang diperlukan (ini akan mengambil dari direktori lokal)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Contoh penggunaan data dari nltk_data_path
stopwords = nltk.corpus.stopwords.words('english')
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
    return ' '.join([word for word in str(teks).split() if word.lower() not in stopwords_indonesian or word.lower() in stopwords_exceptions]) if pd.notna(teks) else ''

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

# Input pengguna untuk ulasan baru
st.subheader('Analisis Ulasan Baru')
user_input = st.text_area('Masukkan ulasan Anda (satu per baris)', '')

# Tambahkan tombol "Tampilkan Hasil Sentimen" yang selalu ada
if st.button('Tampilkan Hasil Sentimen'):
    if user_input:
        # Buat objek stemmer di sini
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        scores = []
        for review in user_input.split('\n'):
            review_cleaned = clean_text(review)
            review_separated = pisahkan_imbuhan_nya(review_cleaned)
            review_normalized = cari_dan_normalisasi_tidak(review_separated, kamus_normalisasi_tidak)
            score, polarity, affected_words = sentiment_lexicon(review_normalized, lexicon_positive, lexicon_negative)
            affected_words_str = ', '.join([f"{word} ({sentiment}: {value})" for word, value, sentiment in affected_words])
            scores.append((review, score, polarity, affected_words_str))

        # Membuat DataFrame untuk menampilkan hasil dalam bentuk tabel
        result_df = pd.DataFrame(scores, columns=['content', 'score', 'sentiment', 'affected_words'])
        result_df['stemmed'] = result_df['content'].apply(lambda x: stemmer.stem(x))
        result_df['lexicon_sentiment'] = result_df['score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

        st.write(result_df)
    else:
        st.warning('Silakan masukkan ulasan untuk dianalisis.')

# Fungsi untuk analisis batch
st.subheader('Analisis Batch Ulasan dari File Excel')
uploaded_file = st.file_uploader("Unggah file Excel dengan kolom 'content'", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    data = data.dropna(subset=['content'])
    data = data.drop_duplicates(subset=['content'])

    # Buat objek stemmer di sini
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Proses data
    data['content'] = data['content'].apply(clean_text)
    data = data[data['content'].apply(lambda x: len(str(x).split()) >= 3)]
    data['tidak_frasa'] = data['content'].apply(lambda x: [f"{w} {x.split()[i+1]}" for i, w in enumerate(x.split()) if w.lower() == 'tidak'] if isinstance(x, str) else [])
    df_terfilter = data[data['tidak_frasa'].apply(bool)]
    df_terfilter['norm_tidak'] = df_terfilter['content'].apply(lambda x: cari_dan_normalisasi_tidak(x, kamus_normalisasi_tidak))
    data.update(df_terfilter)

    kamus_normalisasi = pd.read_excel('kamus/kbba_komplit.xlsx').set_index('non_baku')['baku'].to_dict()
    data['normalisasi'] = data['content'].apply(lambda x: normalisasi(x, kamus_normalisasi))

    stopwords_indonesian = pd.read_excel('kamus/stopword_dictionary.xlsx')['stopword'].tolist()
    stopwords_indonesian.extend(['zok', 'lpk', 'zpk', 'brr', 'jrg', 'vrz', 'fnl', 'ttk', 'sgp', 'an', 'nya',
                                 'bukalapak', 'dana', 'shopee', 'ovo', 'gopay'])
    stopwords_exceptions = {'tidak', 'belakang', 'lama', 'laku', 'malah', 'seperti', 'dan', 'setelah', 'untuk',
                            'sehingga', 'sedangkan', 'meski', 'karena'}
    data['clean'] = data['normalisasi'].apply(lambda x: hapus_stopwords(x, stopwords_indonesian, stopwords_exceptions))
    
    # Pisahkan imbuhan "-nya" dalam teks
    data['clean'] = data['clean'].apply(pisahkan_imbuhan_nya)

    # Stemming
    data['stemmed'] = data['clean'].apply(stemmer.stem)

    tfidf = TfidfVectorizer(max_features=1000)
    tfidf.fit(data['clean'])
    data['tfidf'] = list(tfidf.transform(data['clean']).toarray())

    data['sentiment_score'] = data['clean'].apply(lambda x: sentiment_lexicon(x, lexicon_positive, lexicon_negative)[0])
    data['sentiment_polarity'] = data['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

    # Tambahkan filter untuk ulasan dengan kata minimal 3 dan hapus ulasan netral
    data = data[data['sentiment_polarity'] != 'neutral']

    # Konversi affected_words menjadi string
    data['affected_words'] = data['clean'].apply(lambda x: sentiment_lexicon(x, lexicon_positive, lexicon_negative)[2])
    data['affected_words'] = data['affected_words'].apply(lambda words: ', '.join([f"{word} ({sentiment}: {value})" for word, value, sentiment in words]))

    # Tampilkan hasil pada aplikasi
    st.subheader('Hasil Sentimen Ulasan:')
    st.write(data[['content', 'sentiment_polarity', 'affected_words']])

    # Visualisasikan distribusi sentimen
    st.subheader('Distribusi Sentimen:')
    sentiment_counts = data['sentiment_polarity'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  # Adjust figure size and DPI
    ax.bar(sentiment_counts.index, sentiment_counts.values, width=0.4)  # Adjust bar width if needed
    ax.set_xlabel('Sentimen', fontsize=12)
    ax.set_ylabel('Jumlah', fontsize=12)
    ax.set_title('Distribusi Sentimen', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick parameters
    st.pyplot(fig)

    # Simpan model
    model = {'lexicon_positive': lexicon_positive, 'lexicon_negative': lexicon_negative}
    save_model(model, 'sentiment_model.pkl')
    st.success('Model berhasil disimpan ke sentiment_model.pkl')
else:
    st.warning('Silakan masukkan ulasan atau unggah file Excel untuk analisis.')
