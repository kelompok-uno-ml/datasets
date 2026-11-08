import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def latih_model_xgboost(nama_file):
    print(f"Membaca dataset bersih: {nama_file} ...")
    try:
        df = pd.read_csv(nama_file)
    except FileNotFoundError:
        print("File tidak ditemukan. Pastikan proses preprocessing sudah selesai.")
        return

    # Menentukan Fitur (X) dan Target (y)
    # Target prediksi kita adalah Views_Numeric
    X = df[['Jumlah_Tagar', 'Skor_Sentimen']]
    y = df['Views_Numeric']
    
    # Karena data awal mungkin masih banyak yang 0, kita beri peringatan jika datanya kurang variatif
    if y.sum() == 0:
        print("\n⚠️ PERINGATAN: Semua nilai Views_Numeric adalah 0.")
        print("Model ML tidak bisa belajar dari data yang tidak memiliki variasi.")
        print("Solusi: Lakukan scraping ulang dengan kode yang sudah ditambahkan 'wait_selector' agar angka views asli masuk ke CSV.")

    print("\nMembagi data menjadi Data Latih (80%) dan Data Uji (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melatih model XGBoost Regressor...")
    # Konfigurasi XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Melakukan prediksi pada Data Uji...")
    y_pred = model.predict(X_test)

    # Evaluasi berdasarkan skema metrik di proposal (III.D)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("-" * 50)
    print("=== HASIL EVALUASI MODEL XGBOOST ===")
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"R-squared (R2) Score      : {r2:.4f}")
    print("-" * 50)

# Jalankan pelatihan menggunakan dataset yang sudah dibersihkan
latih_model_xgboost("dataset_bersih_ML.csv")