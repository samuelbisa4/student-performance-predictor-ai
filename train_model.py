
# ==========================================
# STUDENT PERFORMANCE PREDICTOR AI
# MACHINE LEARNING - LINEAR REGRESSION
# ==========================================

import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1. Membaca dataset
BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "student_performance_dataset.csv")

print("Dataset berhasil dimuat!")
print("Jumlah data:", len(df))


# 2. Menentukan fitur (input AI)
features = [
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_average",
    "previous_exam_score",
    "sleep_hours"
]

X = df[features]

# 3. Menentukan target (output AI)
y = df["final_score"]


# 4. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nPembagian Dataset:")
print("Data Training:", len(X_train))
print("Data Testing:", len(X_test))


# 5. Membuat model AI
model = LinearRegression()

# 6. Melatih model
model.fit(X_train, y_train)

print("\nModel AI berhasil dilatih!")


# 7. Menguji model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(y_test, predictions) ** 0.5

r2 = r2_score(y_test, predictions)


# 8. Menampilkan hasil evaluasi
print("\n================================")
print("       HASIL EVALUASI AI")
print("================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("================================")


# 9. Menyimpan model AI
model_data = {
    "model": model,
    "features": features
}

joblib.dump(
    model_data,
    BASE_DIR / "student_performance_model.pkl"
)

print("\nModel AI berhasil disimpan!")
print("File: student_performance_model.pkl")
