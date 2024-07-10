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

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Helper functions
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
        if lexicon_positive is not None and word in lexicon_positive:
            score += lexicon_positive[word]
            affected_words.append((word, lexicon_positive[word], 'positive'))
        elif lexicon_negative is not None and word in lexicon_negative:
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

def evaluate_svm_kfold(X, y, k=5, kernel='linear', C=1.0, gamma='scale'):
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

# Streamlit app configuration
st.set_page_config(
    page_title="Analisis Sentimen Ulasan Aplikasi MPStore",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS for styling
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

# Add logo and title
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

# Load model if available
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

# User input for new reviews
st.subheader('Analisis Ulasan Baru')
user_input = st.text_area('Masukkan ulasan Anda (satu per baris)', '')

# Add "Tampilkan Hasil Sentimen" button
if st.button('Tampilkan Hasil Sentimen'):
    if user_input:
        scores = []
        for review in user_input.split('\n'):
            score, polarity, affected_words = sentiment_lexicon(clean_text(review), lexicon_positive, lexicon_negative)
            scores.append((review, score, polarity, affected_words))

        st.subheader('Hasil Sentimen :')
        for review, score, polarity, affected_words in scores:
            st.write(f"Sentimen: {polarity}")
            st.write(data[['content', 'Score_polarity', 'Polarity','Sentiment']])
    else:
        st.warning("Harap masukkan ulasan sebelum menekan tombol 'Tampilkan Hasil Sentimen'.")

# Upload Excel file for reviews
uploaded_file = st.file_uploader("Unggah file Excel dengan ulasan", type=["xlsx"])

# Process user input or uploaded file
if user_input or uploaded_file:
    if user_input:
        reviews = user_input.split('\n')
        data = pd.DataFrame(reviews, columns=['content'])
    else:
        data = pd.read_excel(uploaded_file)
        if len(data.columns) == 1:
            data.columns = ['content']
        else:
            st.error('File Excel harus memiliki satu kolom dengan ulasan.')

    if st.button('Analisis Sentimen'):
        # Clean and preprocess text
        data['content'] = data['content'].astype(str).apply(clean_text)

        kamus_normalisasi = pd.read_excel('kamus/kbba_komplit.xlsx').set_index('non_baku')['baku'].to_dict()
        data['normalisasi'] = data['content'].apply(lambda x: normalisasi(x, kamus_normalisasi))

        stopwords_indonesian = pd.read_excel('kamus/stopword_dictionary.xlsx')['stopword'].tolist()
        stopwords_indonesian.extend(['zok', 'lpk', 'zpk', 'brr', 'jrg', 'vrz', 'fnl', 'ttk', 'sgp', 'an', 'nya',
                                     'bukalapak', 'dana', 'shopee', 'ovo', 'gopay'])
        stopwords_exceptions = {'tidak', 'belakang', 'lama', 'baik', 'bagaimana'}
        data['stopword'] = data['normalisasi'].apply(lambda x: hapus_stopwords(x, stopwords_indonesian, stopwords_exceptions))

        data['tokenize'] = data['stopword'].apply(lambda x: nltk.word_tokenize(x) if isinstance(x, str) else [])

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        tqdm.pandas()
        data['stemmed'] = data['tokenize'].progress_apply(lambda x: [stemmer.stem(word) for word in x])

        data['stemmed'] = data['stemmed'].apply(clean_and_combine)

        data[['Score', 'Polarity', 'affected_words']] = data['stemmed'].progress_apply(
            lambda x: pd.Series(sentiment_lexicon(x, lexicon_positive, lexicon_negative))
        )

        sentiment_mapping_dict = {'positive': 1, 'negative': -1, 'neutral': 0}
        data['Sentiment'] = data['Polarity'].map(sentiment_mapping_dict)

        st.subheader('Distribusi Sentimen')
        sentiment_counts = data['Polarity'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        ax.axis('equal')
        st.pyplot(fig)

        st.subheader('Hasil Klasifikasi')
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['stemmed'])
        y = data['Sentiment']

        with st.spinner('Melatih dan mengevaluasi model SVM...'):
            cv_scores, splits, svm_model = evaluate_svm_kfold(X, y, k=5, kernel='linear', C=1.0, gamma='scale')
            avg_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
            st.write('Rata-rata hasil k-fold cross-validation:')
            st.write(pd.DataFrame(avg_scores, index=[0]))

            cm_sum = np.zeros((3, 3), dtype=int)
            for train_index, test_index in splits:
                cm = process_fold(pd.DataFrame(X.toarray()), pd.Series(y), train_index, test_index, svm_model)
                cm_sum += cm

            cm_display = ConfusionMatrixDisplay(cm_sum, display_labels=['Negatif', 'Netral', 'Positif'])
            cm_display.plot(cmap='Blues')
            st.pyplot(plt.gcf())

        model_save_path = 'sentiment_model.pkl'
        model_data = {'svm_model': svm_model, 'vectorizer': vectorizer, 'lexicon_positive': lexicon_positive, 'lexicon_negative': lexicon_negative}
        save_model(model_data, model_save_path)
        st.success(f'Model SVM disimpan ke {model_save_path}')

# Additional functions for specific text normalization cases
def cari_dan_normalisasi_tidak(teks, kamus):
    if isinstance(teks, str):
        kata_kata = teks.split()
        hasil = []
        i = 0
        while i < len(kata_kata):
            if kata_kata[i].lower() == 'tidak' and i + 1 < len(kata_kata):
                frasa = f"{kata_kata[i]} {kata_kata[i+1]}"
                hasil.append(kamus.get(frasa, frasa))
                i += 2  # Skip the word after "tidak"
            else:
                hasil.append(kata_kata[i])
                i += 1
        return ' '.join(hasil)
    return teks

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

# Load normalization dictionary for "tidak"
kamus_normalisasi_tidak = pd.read_excel('kamus/kamus_tidak.xlsx').set_index('frasa')['normalisasi'].to_dict()
