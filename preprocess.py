import pandas as pd
import re
from textblob import TextBlob

# 1. Fungsi Transformasi Angka TikTok (K/M/B ke Numerik)
def konversi_angka_tiktok(nilai):
    if pd.isna(nilai) or nilai == '0':
        return 0
    nilai = str(nilai).upper().replace(',', '.')
    try:
        if 'K' in nilai:
            return int(float(nilai.replace('K', '')) * 1_000)
        elif 'M' in nilai:
            return int(float(nilai.replace('M', '')) * 1_000_000)
        elif 'B' in nilai:
            return int(float(nilai.replace('B', '')) * 1_000_000_000)
        else:
            return int(float(nilai))
    except ValueError:
        return 0

# 2. Fungsi Ekstraksi Fitur (Jumlah Tagar)
def hitung_tagar(teks):
    if pd.isna(teks):
        return 0
    return len(re.findall(r'#\w+', str(teks)))

# 3. Fungsi Ekstraksi Fitur (Sentimen Teks)
def analisis_sentimen(teks):
    if pd.isna(teks):
        return 0.0 # Netral jika kosong
    # Mengembalikan nilai polaritas: -1 (Negatif) hingga 1 (Positif)
    return TextBlob(str(teks)).sentiment.polarity

def jalankan_preprocessing(nama_file_input, nama_file_output):
    print(f"Membaca dataset mentah: {nama_file_input} ...")
    try:
        df = pd.read_csv(nama_file_input)
    except FileNotFoundError:
        print("File tidak ditemukan. Pastikan nama file sesuai dengan hasil scraping sebelumnya.")
        return

    # A. Pembersihan Data (Cleaning & Missing Values)
    print("Menghapus data kosong dan duplikat...")
    df = df.dropna(subset=['URL_Video']) # Hapus baris jika tidak ada URL
    df = df.drop_duplicates(subset=['URL_Video'], keep='first')
    
    # Isi caption kosong dengan string kosong agar tidak error saat diekstrak
    df['Caption_Mentah'] = df['Caption_Mentah'].fillna("")

    # B. Ekstraksi Fitur Baru
    print("Mengekstrak fitur Jumlah Tagar dan Sentimen Teks...")
    df['Jumlah_Tagar'] = df['Caption_Mentah'].apply(hitung_tagar)
    df['Skor_Sentimen'] = df['Caption_Mentah'].apply(analisis_sentimen)

    # C. Transformasi Data (Format XGBoost)
    print("Mengonversi skor interaksi ke format numerik XGBoost...")
    # Sesuaikan nama kolom dengan output scraper terakhirmu
    if 'Views_Skor_Interaksi' in df.columns:
        df['Views_Numeric'] = df['Views_Skor_Interaksi'].apply(konversi_angka_tiktok)
    
    if 'Skor_Likes' in df.columns:
        df['Likes_Numeric'] = df['Skor_Likes'].apply(konversi_angka_tiktok)
        df['Comments_Numeric'] = df['Skor_Comments'].apply(konversi_angka_tiktok)
        df['Shares_Numeric'] = df['Skor_Shares'].apply(konversi_angka_tiktok)

    # D. Menyimpan Dataset Bersih
    df.to_csv(nama_file_output, index=False, encoding='utf-8')
    print(f"\n✅ Proses selesai! Dataset bersih untuk Machine Learning disimpan di: {nama_file_output}")
    
    print("\nPratinjau Data Bersih:")
    print(df.head())

# Silakan ganti nama file input ini dengan nama file CSV hasil scraping-mu
file_input = "dataset_mentah_ML_machinelearning.csv" 
file_output = "dataset_bersih_ML.csv"

jalankan_preprocessing(file_input, file_output)