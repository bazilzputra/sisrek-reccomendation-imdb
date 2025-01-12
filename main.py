import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Menampilkan judul aplikasi
st.title("Sistem Rekomendasi Film IMDb")

# Mengunggah file dataset
uploaded_file = st.file_uploader("Unggah file dataset IMDb (CSV)", type=["csv"])

if uploaded_file:
    # Membaca dataset yang diunggah
    df = pd.read_csv(uploaded_file)
    st.write("Dataset IMDb:")
    st.dataframe(df.head())

    # Menampilkan informasi tentang dataset
    st.write("Jumlah baris dan kolom:", df.shape)
    st.write("Kolom pada dataset:", df.columns)

    # Preprocessing dataset
    df['IMDb Rating'] = pd.to_numeric(df['IMDb Rating'], errors='coerce')
    df['IMDb Rating'] = df['IMDb Rating'].fillna(df['IMDb Rating'].mean())
    df['Age Rating'] = df['Age Rating'].fillna("Unknown")
    df.dropna(subset=["Title"], inplace=True)

    st.write("Dataset setelah preprocessing:")
    st.dataframe(df.head())

    # Mengubah dataset ke format yang sesuai dengan Surprise
    df_surprise = df.rename(columns={
        "Age Rating": "user_id",
        "Title": "item_id",
        "IMDb Rating": "rating"
    })

    required_columns = ["user_id", "item_id", "rating"]
    reader = Reader(rating_scale=(df_surprise["rating"].min(), df_surprise["rating"].max()))
    data = Dataset.load_from_df(df_surprise[required_columns], reader)

    # Membagi data menjadi trainset dan testset
    trainset, testset = train_test_split(data, test_size=0.25)

    # Menggunakan algoritma KNN Basic
    sim_options = {
        "name": "cosine",
        "user_based": True,
    }
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)

    # Evaluasi model
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    st.write(f"RMSE Model: {rmse}")

    # Input untuk rekomendasi
    st.write("### Sistem Rekomendasi Film")
    user_input = st.text_input("Masukkan Target Usia (Age Rating):")
    movie_input = st.text_input("Masukkan Judul Film (misalnya 'The Dark Knight'):")
    recommend_button = st.button("Lihat Rekomendasi")

    # Menampilkan rekomendasi berdasarkan input pengguna
    if recommend_button:
        if user_input and movie_input:
            pred = algo.predict(uid=user_input, iid=movie_input)
            st.write(f"Prediksi Rating untuk {movie_input} oleh {user_input}: {pred.est}")
        else:
            st.warning("Silakan masukkan Age Rating dan Judul Film untuk mendapatkan rekomendasi!")
