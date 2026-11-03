
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. KONFIGURASI
# =====================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
DATASET_PATH = BASE_DIR / "student_performance_dataset.csv"


# =====================================================
# 2. DESAIN APLIKASI
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --green: #087A52;
    --dark-green: #075B3F;
    --light-green: #E8F4EC;
    --background: #F4F7F5;
    --text: #1D3026;
    --muted: #53675B;
    --border: #DCE6DF;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
}

p, li, label {
    color: var(--muted);
}

hr {
    border-color: var(--border);
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #344B3D !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #263D31 !important;
    font-weight: 600 !important;
}

/* HEADER */

.hero {
    background: linear-gradient(115deg, #075B3F, #0B8A5D);
    padding: 32px 36px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(7, 91, 63, 0.13);
}

.hero-kicker {
    color: #DDF7E8 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.hero-title {
    color: #FFFFFF !important;
    font-family: 'Manrope', sans-serif;
    font-size: 30px;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.hero-desc {
    color: #F1FFF6 !important;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
    max-width: 800px;
}

/* SECTION */

.section-kicker {
    color: #087A52 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.4px;
    margin-bottom: 6px;
}

.section-desc {
    color: #53675B !important;
    font-size: 13px;
    margin-bottom: 16px;
}

/* KPI CARDS */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(20, 60, 38, 0.035);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #53675B !important;
    font-weight: 700 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #123D2B !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800 !important;
}

/* FORM */

div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
}

div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label {
    color: #263D31 !important;
    font-weight: 700 !important;
}

.stNumberInput input {
    color: #172B22 !important;
    background: #FFFFFF !important;
    border: 1px solid #BFCFC4 !important;
    border-radius: 9px !important;
}

div[data-testid="stCaptionContainer"] p {
    color: #53675B !important;
    font-size: 12px;
    line-height: 1.6;
}

/* BUTTON */

.stButton > button,
div[data-testid="stFormSubmitButton"] button,
.stDownloadButton > button {
    background: #087A52 !important;
    color: #FFFFFF !important;
    border: 1px solid #087A52 !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: 700 !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
.stDownloadButton > button:hover {
    background: #075B3F !important;
    color: #FFFFFF !important;
    border-color: #075B3F !important;
}

.stButton > button p,
div[data-testid="stFormSubmitButton"] button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}

/* ALERT DAN EXPANDER */

[data-testid="stAlert"] {
    border-radius: 11px;
}

[data-testid="stAlert"] p {
    color: #263D31 !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 11px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #263D31 !important;
    font-weight: 600 !important;
}

/* TABEL */

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* PROGRESS */

div[data-testid="stProgress"] > div {
    background: #DCE9E0;
    border-radius: 20px;
}

div[data-testid="stProgress"] > div > div {
    background: #087A52;
    border-radius: 20px;
}

/* FOOTER */

.footer {
    color: #53675B !important;
    text-align: center;
    font-size: 12px;
    line-height: 1.9;
    padding: 25px 0;
}

.footer strong {
    color: #075B3F !important;
}

/* RESPONSIVE */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 24px;
    }

    .hero-title {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. MEMUAT MODEL DAN DATASET
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


if not MODEL_PATH.exists():
    st.error("File student_performance_model.pkl tidak ditemukan.")
    st.stop()

if not DATASET_PATH.exists():
    st.error("File student_performance_dataset.csv tidak ditemukan.")
    st.stop()

try:
    model_data = load_model()
    df = load_dataset()

    model = model_data["model"]
    features = model_data["features"]

except Exception as error:
    st.error(f"Gagal memuat model atau dataset: {error}")
    st.stop()


required_columns = [
    "student_id",
    "study_hours_per_day",
    "attendance_percentage",
    "assignment_average",
    "previous_exam_score",
    "sleep_hours",
    "final_score",
    "performance_category"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Kolom dataset tidak ditemukan: "
        + ", ".join(missing_columns)
    )
    st.stop()


# =====================================================
# 4. STATISTIK DATASET
# =====================================================

total_students = len(df)

average_score = df["final_score"].mean()

average_attendance = df["attendance_percentage"].mean()

average_study = df["study_hours_per_day"].mean()

category_order = ["Rendah", "Sedang", "Tinggi"]

category_counts = (
    df["performance_category"]
    .astype(str)
    .value_counts()
    .reindex(category_order, fill_value=0)
)


# =====================================================
# 5. FUNGSI TAMPILAN
# =====================================================

def show_hero(kicker, title, description):

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_section(title, description):

    st.markdown(
        f"""
        <div class="section-kicker">{title}</div>
        <div class="section-desc">{description}</div>
        """,
        unsafe_allow_html=True
    )


def get_category(score):

    if score < 60:
        return "Rendah"

    elif score < 80:
        return "Sedang"

    return "Tinggi"


def get_recommendations(
    study,
    attendance,
    assignment,
    previous_exam,
    sleep
):

    suggestions = []

    if study < 2:
        suggestions.append(
            "Waktu belajar masih singkat. "
            "Buat jadwal belajar harian yang teratur."
        )

    if attendance < 80:
        suggestions.append(
            "Kehadiran perlu ditingkatkan agar siswa "
            "tidak tertinggal materi pembelajaran."
        )

    if assignment < 70:
        suggestions.append(
            "Tinjau kembali materi dan tingkatkan "
            "ketelitian dalam menyelesaikan tugas."
        )

    if previous_exam < 60:
        suggestions.append(
            "Pelajari kembali materi ujian sebelumnya "
            "dan identifikasi topik yang belum dipahami."
        )

    if sleep < 6:
        suggestions.append(
            "Perhatikan waktu istirahat dan usahakan "
            "menjaga pola tidur yang cukup."
        )

    if not suggestions:
        suggestions.append(
            "Pertahankan kebiasaan belajar yang konsisten "
            "dan evaluasi perkembangan secara berkala."
        )

    return suggestions


# =====================================================
# 6. SIDEBAR NAVIGASI
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:#075B3F;
        ">
            🎓 Akademik
        </div>

        <div style="
            font-size:10px;
            font-weight:800;
            letter-spacing:1.3px;
            color:#53675B;
        ">
            STUDENT PERFORMANCE ANALYTICS
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.divider()

    st.markdown("**MENU UTAMA**")

    page = st.radio(
        "Pilih halaman",
        [
            "Dashboard",
            "Prediksi Siswa",
            "Tentang Aplikasi"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**INFORMASI MODEL**")

    st.caption("Algoritma")
    st.write("Linear Regression")

    st.caption("Metode")
    st.write("Supervised Learning")

    st.caption("Data pelatihan")
    st.write(f"{total_students} data siswa")

    st.caption("Versi aplikasi: 1.0")


# =====================================================
# 7. HALAMAN DASHBOARD
# =====================================================

if page == "Dashboard":

    show_hero(
        "STUDENT PERFORMANCE ANALYTICS",
        "Dashboard Akademik",
        "Pantau gambaran performa akademik siswa melalui "
        "data kebiasaan belajar, kehadiran, dan pencapaian pembelajaran."
    )

    show_section(
        "RINGKASAN AKADEMIK",
        "Gambaran umum berdasarkan seluruh data siswa dalam dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Siswa",
        f"{total_students:,}".replace(",", ".")
    )

    c2.metric(
        "Rata-rata Nilai",
        f"{average_score:.1f} / 100"
    )

    c3.metric(
        "Rata-rata Kehadiran",
        f"{average_attendance:.1f}%"
    )

    c4.metric(
        "Rata-rata Belajar",
        f"{average_study:.1f} jam"
    )

    st.write("")
    st.divider()

    show_section(
        "ANALISIS PERFORMA SISWA",
        "Distribusi kategori performa dan gambaran nilai berdasarkan durasi belajar."
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:

        with st.container(border=True):

            st.subheader("Distribusi Kategori Performa")

            st.caption(
                "Jumlah siswa pada setiap kategori."
            )

            category_df = (
                category_counts
                .rename_axis("Kategori")
                .reset_index(name="Jumlah Siswa")
            )

            st.bar_chart(
                category_df.set_index("Kategori"),
                color="#087A52"
            )

            st.write(
                f"Rendah: **{int(category_counts['Rendah'])} siswa**"
            )

            st.write(
                f"Sedang: **{int(category_counts['Sedang'])} siswa**"
            )

            st.write(
                f"Tinggi: **{int(category_counts['Tinggi'])} siswa**"
            )

    with chart_right:

        with st.container(border=True):

            st.subheader(
                "Rata-rata Nilai Berdasarkan Jam Belajar"
            )

            st.caption(
                "Ringkasan nilai akhir untuk setiap durasi belajar."
            )

            study_summary = (
                df.groupby("study_hours_per_day")["final_score"]
                .mean()
                .round(1)
                .rename("Rata-rata Nilai")
            )

            st.line_chart(
                study_summary,
                color="#087A52"
            )

    st.write("")
    st.divider()

    show_section(
        "DATA SISWA",
        "Sepuluh baris pertama dari dataset yang digunakan dalam proyek ini."
    )

    table = df[required_columns].head(10).copy()

    table.columns = [
        "ID Siswa",
        "Jam Belajar",
        "Kehadiran (%)",
        "Rata-rata Tugas",
        "Nilai Ujian Sebelumnya",
        "Jam Tidur",
        "Nilai Akhir",
        "Kategori Performa"
    ]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 8. HALAMAN PREDIKSI SISWA
# =====================================================

elif page == "Prediksi Siswa":

    show_hero(
        "ACADEMIC PERFORMANCE ASSESSMENT",
        "Prediksi Performa Siswa",
        "Masukkan informasi kebiasaan belajar dan pencapaian "
        "akademik untuk memperoleh estimasi nilai akhir dari model Machine Learning."
    )

    show_section(
        "FORMULIR DATA SISWA",
        "Isi seluruh indikator sesuai kondisi siswa."
    )

    with st.form("prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:

            st.subheader("Kebiasaan Belajar")

            study_hours = st.number_input(
                "Waktu Belajar per Hari (jam)",
                min_value=0.5,
                max_value=8.0,
                value=3.0,
                step=0.5,
                help="Rata-rata waktu belajar siswa dalam satu hari."
            )

            st.caption(
                "Contoh: 3 berarti siswa belajar sekitar 3 jam per hari."
            )

            attendance = st.number_input(
                "Kehadiran Siswa (%)",
                min_value=50,
                max_value=100,
                value=85,
                step=1,
                help="Persentase kehadiran siswa selama pembelajaran."
            )

            st.caption(
                "Contoh: 85 berarti tingkat kehadiran sebesar 85 persen."
            )

            sleep_hours = st.number_input(
                "Durasi Tidur per Hari (jam)",
                min_value=4.0,
                max_value=10.0,
                value=7.0,
                step=0.5,
                help="Rata-rata jumlah jam tidur siswa setiap hari."
            )

            st.caption(
                "Masukkan rata-rata durasi tidur dalam jam."
            )

        with right:

            st.subheader("Pencapaian Akademik")

            assignment_average = st.number_input(
                "Rata-rata Nilai Tugas",
                min_value=30,
                max_value=100,
                value=75,
                step=1,
                help="Rata-rata nilai tugas siswa."
            )

            st.caption(
                "Gunakan skala nilai 30 sampai 100."
            )

            previous_exam = st.number_input(
                "Nilai Ujian Sebelumnya",
                min_value=20,
                max_value=100,
                value=75,
                step=1,
                help="Nilai yang diperoleh pada ujian sebelumnya."
            )

            st.caption(
                "Gunakan skala nilai 20 sampai 100."
            )

            st.write("")

            st.info(
                "Periksa kembali angka yang dimasukkan. "
                "Model akan menggunakan kelima indikator "
                "untuk memperkirakan nilai akhir."
            )

        st.write("")

        submitted = st.form_submit_button(
            "Hitung Prediksi Performa",
            use_container_width=True
        )

    # PROSES PREDIKSI

    if submitted:

        input_values = {
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "assignment_average": assignment_average,
            "previous_exam_score": previous_exam,
            "sleep_hours": sleep_hours
        }

        try:

            input_df = pd.DataFrame([input_values])

            input_df = input_df.reindex(
                columns=features
            )

            if input_df.isnull().any().any():

                st.error(
                    "Fitur pada model tidak cocok dengan data input. "
                    "Periksa kembali train_model.py."
                )

            else:

                score = float(
                    model.predict(input_df)[0]
                )

                score = max(
                    0.0,
                    min(100.0, score)
                )

                category = get_category(score)

                recommendations = get_recommendations(
                    study_hours,
                    attendance,
                    assignment_average,
                    previous_exam,
                    sleep_hours
                )

                st.session_state["prediction_result"] = {
                    "score": score,
                    "category": category,
                    "recommendations": recommendations
                }

                if "prediction_history" not in st.session_state:

                    st.session_state["prediction_history"] = []

                st.session_state["prediction_history"].insert(
                    0,
                    {
                        "Waktu": pd.Timestamp.now().strftime(
                            "%d-%m-%Y %H:%M"
                        ),
                        "Nilai Prediksi": round(score, 2),
                        "Kategori": category
                    }
                )

        except Exception as error:

            st.error(
                f"Prediksi gagal dilakukan: {error}"
            )

    # HASIL PREDIKSI

    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]

        score = result["score"]
        category = result["category"]

        st.write("")
        st.divider()

        show_section(
            "HASIL ANALISIS",
            "Berikut perkiraan nilai akhir berdasarkan informasi yang telah dimasukkan."
        )

        result_left, result_right = st.columns(2)

        with result_left:

            st.metric(
                "Perkiraan Nilai Akhir",
                f"{score:.2f} / 100"
            )

        with result_right:

            st.metric(
                "Kategori Performa",
                category
            )

        st.progress(
            int(round(score)),
            text=f"Perkiraan nilai: {score:.2f} dari 100"
        )

        if category == "Rendah":

            st.warning(
                "Perkiraan nilai berada pada kategori rendah. "
                "Siswa dapat memerlukan pendampingan belajar tambahan."
            )

        elif category == "Sedang":

            st.info(
                "Perkiraan nilai berada pada kategori sedang. "
                "Konsistensi belajar dan pemahaman materi "
                "dapat terus ditingkatkan."
            )

        else:

            st.success(
                "Perkiraan nilai berada pada kategori tinggi. "
                "Pertahankan kebiasaan belajar yang baik."
            )

        st.subheader("Saran yang Dapat Dipertimbangkan")

        for number, suggestion in enumerate(
            result["recommendations"],
            start=1
        ):

            st.write(f"**{number}.** {suggestion}")

        st.caption(
            "Catatan: hasil merupakan estimasi model, bukan "
            "nilai akhir yang pasti. Rekomendasi adalah saran "
            "umum berdasarkan nilai input."
        )

    # RIWAYAT PREDIKSI

    if st.session_state.get("prediction_history"):

        st.write("")
        st.divider()

        st.subheader("Riwayat Prediksi pada Sesi Ini")

        history_df = pd.DataFrame(
            st.session_state["prediction_history"]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Unduh Riwayat Prediksi (CSV)",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="riwayat_prediksi.csv",
            mime="text/csv"
        )


# =====================================================
# 9. HALAMAN TENTANG APLIKASI
# =====================================================

elif page == "Tentang Aplikasi":

    show_hero(
        "PROJECT INFORMATION",
        "Tentang Aplikasi",
        "Kenali tujuan, indikator, metode Machine Learning, "
        "dan keterbatasan Student Performance Predictor."
    )

    st.subheader(
        "Apa itu Student Performance Predictor?"
    )

    st.write(
        "Student Performance Predictor adalah aplikasi "
        "demonstrasi Machine Learning untuk memperkirakan "
        "nilai akhir siswa berdasarkan kebiasaan belajar "
        "dan indikator pencapaian akademik."
    )

    st.divider()

    show_section(
        "LIMA INDIKATOR INPUT",
        "Informasi yang digunakan model untuk menghasilkan estimasi nilai."
    )

    feature_left, feature_right = st.columns(2)

    with feature_left:

        with st.container(border=True):

            st.markdown("**1. Waktu Belajar**")

            st.write(
                "Rata-rata waktu belajar siswa setiap hari."
            )

        with st.container(border=True):

            st.markdown("**2. Kehadiran**")

            st.write(
                "Persentase kehadiran siswa dalam pembelajaran."
            )

        with st.container(border=True):

            st.markdown("**3. Rata-rata Nilai Tugas**")

            st.write(
                "Nilai rata-rata tugas yang telah dikerjakan."
            )

    with feature_right:

        with st.container(border=True):

            st.markdown("**4. Nilai Ujian Sebelumnya**")

            st.write(
                "Nilai ujian sebelumnya sebagai salah satu indikator akademik."
            )

        with st.container(border=True):

            st.markdown("**5. Durasi Tidur**")

            st.write(
                "Rata-rata jumlah jam tidur siswa per hari."
            )

    st.divider()

    show_section(
        "METODE MACHINE LEARNING",
        "Penjelasan singkat mengenai algoritma yang digunakan."
    )

    with st.container(border=True):

        st.subheader("Linear Regression")

        st.write(
            "Linear Regression merupakan algoritma pembelajaran "
            "terawasi yang digunakan untuk memperkirakan nilai "
            "numerik berdasarkan pola hubungan antara variabel "
            "input dan target."
        )

    st.subheader("Alur Kerja Aplikasi")

    step1, step2, step3, step4 = st.columns(4)

    step1.markdown("### 01")
    step1.markdown("**Input Data**")
    step1.caption("Pengguna mengisi lima indikator.")

    step2.markdown("### 02")
    step2.markdown("**Pemrosesan**")
    step2.caption("Data disusun sesuai fitur model.")

    step3.markdown("### 03")
    step3.markdown("**Prediksi**")
    step3.caption("Model menghitung estimasi nilai.")

    step4.markdown("### 04")
    step4.markdown("**Hasil**")
    step4.caption("Nilai dan kategori ditampilkan.")

    st.divider()

    st.subheader("Keterbatasan Aplikasi")

    st.warning(
        "Dataset proyek ini merupakan data sintetis sebanyak "
        f"{total_students} baris, bukan data siswa sungguhan. "
        "Hasil prediksi hanya untuk demonstrasi dan pembelajaran, "
        "belum divalidasi untuk digunakan sebagai dasar keputusan akademik nyata."
    )

    st.write(
        "**Teknologi:** Python, Pandas, Scikit-learn, "
        "Joblib, dan Streamlit."
    )


# =====================================================
# 10. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <strong>STUDENT PERFORMANCE PREDICTOR</strong><br>
    Academic Analytics & Machine Learning Project<br>
    Dikembangkan menggunakan Python dan Streamlit · 2026
</div>
""", unsafe_allow_html=True)
