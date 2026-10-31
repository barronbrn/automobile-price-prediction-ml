# 🚗 Automobile Price Prediction: Multivariate Regression & Explainable AI (SHAP)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-00A389?style=for-the-badge)](https://shap.readthedocs.io)
[![Certification](https://img.shields.io/badge/Standard-BNSP%20Data%20Science-blue?style=for-the-badge)](https://bnsp.go.id)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> 🏆 **Proyek Uji Kompetensi BNSP** — Skema Sertifikasi *Associate Data Scientist*, LSP Informatika  
> 🔬 **Fokus:** *Predictive Pricing Analytics, Multivariate Regression, Explainable AI (XAI), Automated Machine Learning (AutoML)*

---

## 📋 Deskripsi Proyek

Penentuan harga pasar kendaraan (*automobile valuation*) yang akurat merupakan tantangan mendasar dalam industri otomotif dan *dealership*. Valuasi yang tepat membantu bisnis memaksimalkan margin keuntungan, mempercepat perputaran inventaris, dan memberikan harga yang adil bagi konsumen.

Proyek ini membangun model regresi prediktif berbasis *Machine Learning* untuk memprediksi harga mobil berdasarkan spesifikasi teknis dan karakteristik kendaraan. Proyek ini dikembangkan secara end-to-end mengacu pada **Standar Kompetensi Kerja Nasional Indonesia (SKKNI) Bidang Data Science**, dilengkapi dengan interpretasi model menggunakan **Explainable AI (SHAP)** untuk menganalisis faktor spesifikasi yang paling memengaruhi harga mobil.

---

## 📊 Dataset & Fitur Utama

Dataset terdiri dari data spesifikasi otomotif komprehensif dengan variabel target kontinu:

| Fitur | Deskripsi | Tipe |
| :--- | :--- | :--- |
| `engine-size` | Kapasitas mesin kendaraan (cc / cubic inches) | Numerik |
| `curb-weight` | Berat kosong kendaraan (lbs) | Numerik |
| `horsepower` | Tenaga kuda kendaraan (hp) | Numerik |
| `highway-mpg` | Konsumsi bahan bakar jalan tol (miles per gallon) | Numerik |
| `city-mpg` | Konsumsi bahan bakar perkotaan (miles per gallon) | Numerik |
| `wheel-base` | Jarak sumbu roda kendaraan (inci) | Numerik |
| `length`, `width`, `height` | Dimensi fisik kendaraan | Numerik |
| `num-of-cylinders` | Jumlah silinder mesin | Kategorikal |
| `fuel-type` | Jenis bahan bakar (gas, diesel) | Kategorikal |
| `aspiration` | Sistem induksi udara (std, turbo) | Kategorikal |
| `drive-wheels` | Sistem penggerak roda (rwd, fwd, 4wd) | Kategorikal |
| **`price`** | **Harga jual kendaraan dalam USD ($)** | **Target Regresi** |

---

## 🏗️ Metodologi & Alur Pengembangan (SKKNI BNSP)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REGRESSION PIPELINE (SKKNI)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] Penelaahan Data (Data Understanding)                                   │
│      └── Analisis tipe data, sebaran distribusi harga, korelasi fitur       │
│                                                                             │
│  [2] Validasi & Pembersihan Data (Data Cleaning)                            │
│      ├── Deteksi & imputasi missing values (karakter khusus '?' diimputasi) │
│      └── Penanganan outlier pada fitur numerik kunci                        │
│                                                                             │
│  [3] Konstruksi & Rekayasa Fitur                                            │
│      ├── One-Hot Encoding fitur kategorikal (drive-wheels, fuel-type, dll)  │
│      └── Standarisasi skala fitur untuk model linier                        │
│                                                                             │
│  [4] Explainable AI (SHAP) untuk Seleksi Fitur                              │
│      └── Mengidentifikasi fitur yang memiliki kontribusi terbesar           │
│                                                                             │
│  [5] Pemodelan Regresi Komparatif                                           │
│      ├── Linear Regression (Baseline)                                       │
│      ├── Random Forest Regressor (Ensemble Bagging)                         │
│      ├── XGBoost Regressor (Gradient Boosting)                              │
│      └── Automated Machine Learning (AutoML MLJAR)                          │
│                                                                             │
│  [6] Evaluasi Metrik Kinerja                                                │
│      └── R-Squared (R²), Mean Absolute Error (MAE), Root Mean Squared Error │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Hasil Evaluasi & Perbandingan Model

Evaluasi model dilakukan pada *unseen test set*:

| Model / Algoritma | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | R² Score (Koefisien Determinasi) |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor** 🥇 | **$835.05** | **$1,257.77** | **0.9800 (98.00%)** |
| **Random Forest Regressor (n=20)** 🥈 | $1,259.17 | $1,781.63 | 0.9598 (95.98%) |
| **Linear Regression (Baseline)** | $1,951.54 | $2,551.37 | 0.9176 (91.76%) |

> 🚀 **Insight Hasil:**  
> - **XGBoost Regressor** mencapai performa luar biasa dengan **R² Score sebesar 0.9800 (98.00%)**, artinya model mampu menjelaskan 98% variasi harga mobil pada data pengujian.
> - Tingkat kesalahan rata-rata (**MAE**) berhasil ditekan hingga **$835.05**, penurunan error lebih dari **57%** dibandingkan model baseline Linear Regression ($1,951.54).

---

## 🔍 Wawasan Explainable AI (SHAP Feature Importance)

Berdasarkan analisis nilai **SHAP (SHapley Additive exPlanations)**:
1. **`engine-size`**: Faktor paling dominan; semakin besar kapasitas mesin, semakin signifikan dorongan kenaikan harga mobil.
2. **`curb-weight`**: Berat kendaraan berkorelasi positif kuat terhadap harga (mencerminkan ukuran kendaraan, material premium, dan stabilitas).
3. **`horsepower`**: Output tenaga mesin menjadi pembeda utama pada segmen kendaraan mewah/sport.
4. **`highway-mpg`**: Memiliki korelasi negatif; mobil dengan konsumsi bahan bakar yang sangat hemat (efisiensi tinggi/mobil ekonomis) umumnya berada pada rentang harga yang lebih terjangkau.

---

## 🗂️ Struktur Direktori

```text
├── 📓 automobile_price_prediction.ipynb  # Notebook analisis regresi end-to-end
├── 📊 automobile21.csv                   # Dataset spesifikasi otomotif
├── 📄 requirements.txt                   # Dependensi pustaka Python
├── 📄 .gitignore                         # File konfigurasi Git
└── 📄 README.md                          # Dokumentasi proyek
```
