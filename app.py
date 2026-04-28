import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("📚 Sistem Rekomendasi Buku")
st.markdown("""
Aplikasi ini menggunakan algoritma **k-Nearest Neighbors (kNN)** dengan *Cosine Similarity* untuk merekomendasikan buku berdasarkan kemiripan rating antar item.
""")

# 1. Load Data & Preprocessing (Cashed agar cepat)
@st.cache_data
def load_and_process_data():
    # Load datasets
    books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1", low_memory=False)
    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")

    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    users.columns = ['userID', 'Location', 'Age']
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    # Filter Users (Minimal 200 rating) & Ratings (Minimal 100 rating)
    counts1 = ratings['userID'].value_counts()
    ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
    
    # Gabungkan Rating dengan Judul Buku
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)

    # Hitung Total Rating per Buku
    book_ratingCount = (combine_book_rating.
         groupby(by = ['bookTitle'])['bookRating'].
         count().reset_index().
         rename(columns = {'bookRating': 'totalRatingCount'})
        )
    
    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, on='bookTitle', how='left')

    # Filter Buku Populer (Threshold >= 50)
    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    # Filter Lokasi USA & Canada (Sesuai Notebook)
    combined = rating_popular_book.merge(users, on='userID', how='left')
    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada", na=False)]
    
    # Drop duplicates dan buat Pivot Table
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID', values='bookRating').fillna(0)
    
    return pivot

# Load data ke session state
try:
    with st.spinner('Memproses data... Mohon tunggu...'):
        pivot_table = load_and_process_data()
        matrix = csr_matrix(pivot_table.values)
    st.success('Data berhasil dimuat!')
except Exception as e:
    st.error(f"Gagal memuat file CSV. Pastikan file dataset tersedia di folder aplikasi. Error: {e}")
    st.stop()

# 2. Build Model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(matrix)

# 3. Sidebar - Pilih Buku
st.sidebar.header("Pilih Judul Buku")
book_list = pivot_table.index.tolist()
selected_book = st.sidebar.selectbox("Ketik atau pilih buku:", book_list)

# 4. Main Page - Tampilan Analisis
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Statistik Data Terfilter")
    st.write(f"Jumlah Buku Populer: {pivot_table.shape[0]}")
    st.write(f"Jumlah User (USA/Canada): {pivot_table.shape[1]}")

with col2:
    # Visualisasi Sederhana Rating Distribution
    st.subheader("Distribusi Rating")
    fig, ax = plt.subplots(figsize=(6, 4))
    # Mengambil sampel data rating non-nol untuk visualisasi
    pivot_table.replace(0, np.nan).stack().plot.hist(bins=10, ax=ax, color='skyblue')
    ax.set_xlabel("Rating")
    st.pyplot(fig)

st.divider()

# 5. Hasil Rekomendasi
if st.button('Berikan Rekomendasi'):
    distances, indices = model_knn.kneighbors(
        pivot_table.loc[selected_book, :].values.reshape(1, -1), 
        n_neighbors=6
    )

    st.subheader(f"Rekomendasi untuk buku: **{selected_book}**")
    
    recom_data = []
    for i in range(1, len(distances.flatten())):
        recom_data.append({
            "Peringkat": i,
            "Judul Buku": pivot_table.index[indices.flatten()[i]],
            "Distance (Cosine)": round(distances.flatten()[i], 4)
        })
    
    st.table(pd.DataFrame(recom_data))
    st.info("Catatan: Semakin rendah nilai *Distance*, semakin mirip buku tersebut dengan buku yang dipilih.")

else:
    st.write("Silakan pilih buku di sidebar dan klik tombol rekomendasi.")