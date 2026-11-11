import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def latih_dan_bandingkan_model(nama_file):
    print(f"Membaca dataset bersih: {nama_file} ...\n")
    try:
        df = pd.read_csv(nama_file)
    except FileNotFoundError:
        print("File tidak ditemukan. Pastikan proses preprocessing sudah selesai.")
        return

    # Menentukan Fitur (X) dan Target (y)
    X = df[['Jumlah_Tagar', 'Skor_Sentimen']]
    y = df['Views_Numeric']
    
    if y.sum() == 0:
        print("⚠️ PERINGATAN: Semua nilai Views_Numeric adalah 0. Scraping ulang dengan data riil diperlukan.\n")

    print("Membagi data menjadi Data Latih (80%) dan Data Uji (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # MODEL 1: XGBOOST (Metode Boosting)
    # ==========================================
    print("\n[1/2] Melatih model XGBoost Regressor...")
    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # ==========================================
    # MODEL 2: RANDOM FOREST (Metode Bagging)
    # ==========================================
    print("[2/2] Melatih model Random Forest Regressor (Model Pembanding)...")
    model_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # ==========================================
    # HASIL EVALUASI DAN PERBANDINGAN
    # ==========================================
    print("\n" + "=" * 50)
    print("HASIL EVALUASI KOMPARATIF ENSEMBLE LEARNING")
    print("=" * 50)
    print("1. XGBOOST (Boosting)")
    print(f"   - Mean Absolute Error (MAE) : {mae_xgb:,.2f}")
    print(f"   - R-squared (R2) Score      : {r2_xgb:.4f}")
    print("-" * 50)
    print("2. RANDOM FOREST (Bagging)")
    print(f"   - Mean Absolute Error (MAE) : {mae_rf:,.2f}")
    print(f"   - R-squared (R2) Score      : {r2_rf:.4f}")
    print("=" * 50)
    
    # Rekomendasi otomatis berdasarkan R2
    pemenang = "XGBoost" if r2_xgb > r2_rf else "Random Forest"
    print(f"💡 Kesimpulan: Untuk dataset ini, model {pemenang} menunjukkan kinerja yang lebih baik.")

# Jalankan pelatihan komparatif
latih_dan_bandingkan_model("dataset_bersih_ML.csv")