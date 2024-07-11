from google_play_scraper import Sort, reviews

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

# Tambahkan bagian untuk scraping ulasan
st.subheader('Scrape Ulasan dari Google Play')
app_id = st.text_input('Masukkan Google Play App ID', '')
num_reviews = st.number_input('Jumlah Ulasan yang Ingin Diambil', min_value=1, max_value=1000, value=100)

if st.button('Scrape Ulasan'):
    if app_id:
        scraped_reviews = []
        for i in tqdm(range(0, num_reviews, 100)):
            scraped_reviews.extend(reviews(
                app_id,
                count=min(num_reviews - len(scraped_reviews), 100),
                sort=Sort.NEWEST
            )[0])
        st.success(f'{len(scraped_reviews)} ulasan berhasil di-scrape')
        reviews_df = pd.DataFrame(scraped_reviews)
        st.write(reviews_df[['content', 'score', 'at']])

        # Simpan ulasan yang di-scrape ke file Excel
        reviews_df.to_excel('scraped_reviews.xlsx', index=False)
        st.success('Ulasan yang di-scrape telah disimpan ke scraped_reviews.xlsx')
    else:
        st.warning('Silakan masukkan Google Play App ID.')
