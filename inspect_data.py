
import pandas as pd
from pathlib import Path

# 1. Membaca dataset
file_path = Path(__file__).parent / "student_performance_dataset.csv"

df = pd.read_csv(file_path)

# 2. Menampilkan informasi dataset
print("====================================")
print("   STUDENT PERFORMANCE DATASET")
print("====================================")

print("\n1. JUMLAH DATA")
print("Jumlah siswa:", len(df))
print("Jumlah kolom:", len(df.columns))

print("\n2. 5 DATA SISWA PERTAMA")
print(df.head())

print("\n3. INFORMASI DATASET")
df.info()

print("\n4. CEK DATA KOSONG")
print(df.isnull().sum())

print("\n5. RATA-RATA NILAI AKHIR")
print(df["final_score"].mean())

print("\n6. DISTRIBUSI KATEGORI PERFORMA")
print(df["performance_category"].value_counts())

print("\n====================================")
print("   DATASET BERHASIL DIBACA!")
print("====================================")
