import os
import nltk
import subprocess

# Tentukan lokasi direktori nltk_data lokal
repo_dir = 'sidang_imam'
nltk_data_path = os.path.join(repo_dir, 'nltk_data')

# Pastikan direktori nltk_data sudah ada
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Tambahkan nltk_data_path ke path nltk
nltk.data.path.append(nltk_data_path)

# Jika Anda ingin menggunakan repository git untuk mengambil data
repo_url = 'https://github.com/imamsyafiihreal/sidang_imam.git'
if not os.path.exists(repo_dir):
    subprocess.run(['git', 'clone', repo_url])

# Fungsi untuk mengunduh corpus nltk
def download_nltk_data():
    try:
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('punkt', download_dir=nltk_data_path)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

# Coba unduh data nltk
download_nltk_data()

# Cek apakah data sudah terunduh
from nltk.corpus import stopwords
stop_words = stopwords.words('english')  # Ganti dengan bahasa yang sesuai

print(f"Jumlah stopwords: {len(stop_words)}")

# Selanjutnya, lanjutkan dengan penggunaan data Anda seperti yang diperlukan
