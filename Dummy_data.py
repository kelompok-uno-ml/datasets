import pandas as pd
import random

# Membaca dataset bersih yang saat ini nilainya 0
df = pd.read_csv("dataset_bersih_ML.csv")

# Menyuntikkan data acak agar model XGBoost bisa belajar mengenali pola
print("Menyuntikkan variasi data buatan untuk pengujian model...")
df['Views_Numeric'] = [random.randint(1000, 5000000) for _ in range(len(df))]
df['Jumlah_Tagar'] = [random.randint(2, 12) for _ in range(len(df))]
df['Skor_Sentimen'] = [random.uniform(-1.0, 1.0) for _ in range(len(df))]

# Menyimpan kembali ke CSV
df.to_csv("dataset_bersih_ML.csv", index=False)
print("✅ Data sintetis berhasil ditambahkan! Silakan jalankan train_model.py kembali.")