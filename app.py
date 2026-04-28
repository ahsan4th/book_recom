import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Book Recommendation System - Upload Mode", layout="wide")

st.title("📚 Sistem Rekomendasi Buku (Upload CSV)")
st.markdown("""
Silakan unggah 3 file CSV utama (**Books**, **Users**, dan **Ratings**) untuk memulai analisis dan sistem rekomendasi.
""")

# --- BAGIAN UPLOAD FILE ---
st.sidebar.header("📁 Upload Dataset")
uploaded_books = st.sidebar.file_uploader("Upload BX-Books.csv", type="csv")
uploaded_users = st.sidebar.file_uploader("Upload BX-Users.csv", type="csv")
uploaded_ratings = st.sidebar.file_uploader("Upload BX-Book-Ratings.csv", type="csv")

# 1. Fungsi Preprocessing (Data Imputation & Cleaning)
@st.cache_data
def process_data(file_books, file_users, file_ratings):
    # Load dengan penanganan error sesuai notebook
    books = pd.read_csv(file_books, sep=';', on_bad_lines='skip', encoding="latin-1", low_memory=False)
    users = pd.read_csv(file_users, sep=';', on_bad_lines='skip', encoding="latin-1")
    ratings = pd.read_csv(file_ratings, sep=';', on_bad_lines='skip', encoding="latin-1")

    # Rename kolom agar konsisten dengan logika notebook
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    users.columns = ['userID', 'Location', 'Age']
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    # Filter Users: Minimal memberikan 200 rating
    counts_user = ratings['userID'].value_counts()
    ratings = ratings[ratings['userID'].isin(counts_user[counts_user >= 200].index)]
    
    # Gabungkan Rating dengan Judul Buku
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    drop_cols = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(drop_cols, axis=1)

    # Hitung Total Rating per Buku
    book_ratingCount = (combine_book_rating.
         groupby(by=['bookTitle'])['bookRating'].
         count().reset_index().
         rename(columns={'bookRating': 'totalRatingCount'})
        )
    
    rating_with_count = combine_book_rating.merge(book_ratingCount, on='bookTitle', how='left')

    # Filter Buku Populer (Threshold >= 50)
    popularity_threshold = 50
    rating_popular = rating_with_count.query('totalRatingCount >= @popularity_threshold')

    # Filter Lokasi: USA & Canada
    combined = rating_popular.merge(users, on='userID', how='left')
    us_canada_rating = combined[combined['Location'].str.contains("usa|canada", na=False)]
    
    # Buat Pivot Table
    us_canada_rating = us_canada_rating.drop_duplicates(['userID', 'bookTitle'])
    pivot = us_canada_rating.pivot(index='bookTitle', columns='userID', values='bookRating').fillna(0)
    
    return pivot

# --- LOGIKA APLIKASI ---
if uploaded_books and uploaded_users and uploaded_ratings:
    try:
        with st.spinner('Sedang memproses file yang diunggah...'):
            pivot_table = process_data(uploaded_books, uploaded_users, uploaded_ratings)
            matrix = csr_matrix(pivot_table.values)
        st.success('Data Berhasil Diproses!')

        # Build Model kNN
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(matrix)

        # Dashboard Visual
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Buku Terfilter", pivot_table.shape[0])
            st.metric("Total User Terfilter", pivot_table.shape[1])
        
        with c2:
            st.subheader("Distribusi Rating (Non-Zero)")
            fig, ax = plt.subplots(figsize=(6, 3))
            pivot_table.replace(0, np.nan).stack().plot.hist(bins=10, ax=ax, color='green')
            st.pyplot(fig)

        st.divider()

        # Rekomendasi
        st.subheader("🔍 Cari Rekomendasi")
        selected_book = st.selectbox("Pilih judul buku yang Anda sukai:", pivot_table.index)

        if st.button('Tampilkan Rekomendasi'):
            distances, indices = model_knn.kneighbors(
                pivot_table.loc[selected_book, :].values.reshape(1, -1), 
                n_neighbors=6
            )

            st.markdown(f"Buku yang mirip dengan **{selected_book}**:")
            
            recom_list = []
            for i in range(1, len(distances.flatten())):
                recom_list.append({
                    "Judul Buku": pivot_table.index[indices.flatten()[i]],
                    "Skor Kemiripan (Distance)": round(distances.flatten()[i], 4)
                })
            
            st.table(pd.DataFrame(recom_list))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.info("Pastikan format kolom di dalam CSV sesuai dengan dataset Book-Crossing.")

else:
    st.warning("Silakan unggah ketiga file CSV (Books, Users, Ratings) melalui sidebar untuk melanjutkan.")
    st.image("https://via.placeholder.com/800x400.png?text=Menunggu+Upload+File+CSV", use_column_width=True)
