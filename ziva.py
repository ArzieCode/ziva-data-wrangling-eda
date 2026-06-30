# ============================================================
# ZIVA — Data Wrangling & Exploratory Data Analysis
# Dataset: IVFNet (citra embrio time-lapse + data klinis tabular)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
import os

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
# Sesuaikan path dengan lokasi dataset hasil unduhan
df = pd.read_csv("data/ivfnet_clinical.csv")
print("Shape awal:", df.shape)
print(df.head())
print(df.info())

# ------------------------------------------------------------
# 2. PEMERIKSAAN KUALITAS DATA AWAL
# ------------------------------------------------------------
missing_pct = df.isnull().mean() * 100
print("\nPersentase missing value per kolom:")
print(missing_pct.sort_values(ascending=False))

# Cek skewness AMH untuk justifikasi pemilihan strategi imputasi
print("\nSkewness AMH:", df["AMH"].skew())

plt.figure(figsize=(6, 4))
sns.histplot(df["AMH"].dropna(), kde=True)
plt.title("Distribusi AMH Sebelum Imputasi")
plt.xlabel("AMH (pmol/L)")
plt.savefig("outputs/amh_distribution_raw.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 3. IMPUTASI MISSING VALUE (AMH & FSH) — MEDIAN PER KELOMPOK USIA
# ------------------------------------------------------------
def age_group(age):
    if age < 30:
        return "<30"
    elif age < 35:
        return "30-34"
    elif age < 40:
        return "35-39"
    else:
        return "40+"

df["age_group"] = df["usia_ibu"].apply(age_group)

for col in ["AMH", "FSH"]:
    df[col] = df.groupby("age_group")[col].transform(
        lambda x: x.fillna(x.median())
    )

print("\nMissing value setelah imputasi:")
print(df[["AMH", "FSH"]].isnull().sum())

# ------------------------------------------------------------
# 4. STANDARDISASI FITUR TABULAR
# ------------------------------------------------------------
numeric_cols = ["usia_ibu", "AMH", "FSH", "BMI", "jumlah_oosit", "riwayat_siklus"]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nContoh data setelah standardisasi:")
print(df_scaled[numeric_cols].describe())

# ------------------------------------------------------------
# 5. PREPROCESSING CITRA
# ------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img

def filter_unusable_images(image_dir, blur_threshold=100.0):
    """Buang citra yang blur berlebihan / over-under exposure."""
    valid_files = []
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # deteksi blur via variance of Laplacian
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        mean_intensity = img.mean()
        if blur_score < blur_threshold:
            continue  # terlalu blur
        if mean_intensity < 15 or mean_intensity > 240:
            continue  # under/over exposure
        valid_files.append(fname)
    return valid_files

# Contoh pemakaian (sesuaikan path):
# valid_images = filter_unusable_images("data/images/raw")
# print(f"Citra valid: {len(valid_images)} dari total file di folder")

# ------------------------------------------------------------
# 6. PENANGANAN CLASS IMBALANCE
# ------------------------------------------------------------
label_counts = df["label_implantasi"].value_counts(normalize=True) * 100
print("\nDistribusi label (%):")
print(label_counts)

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(df["label_implantasi"])
class_weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=df["label_implantasi"]
)
class_weight_dict = dict(zip(classes, class_weights))
print("\nClass weight:", class_weight_dict)

# Augmentasi citra kelas minoritas (ilustrasi pipeline, pakai albumentations)
import albumentations as A

minority_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.5),
])

# ------------------------------------------------------------
# 7. PEMBAGIAN DATASET (STRATIFIED 70/15/15)
# ------------------------------------------------------------
train_df, temp_df = train_test_split(
    df_scaled, test_size=0.30, stratify=df_scaled["label_implantasi"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label_implantasi"], random_state=42
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Proporsi label tiap subset:")
for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(name, subset["label_implantasi"].value_counts(normalize=True).to_dict())

# ============================================================
# 8. EXPLORATORY DATA ANALYSIS
# ============================================================

# --- 8.1 Distribusi Variabel Klinis ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(df["usia_ibu"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("(a) Distribusi Usia Ibu")

sns.histplot(df["AMH"], kde=True, ax=axes[0, 1])
axes[0, 1].set_title("(b) Distribusi AMH")

sns.histplot(df["FSH"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("(c) Distribusi FSH")

sns.histplot(df["BMI"], kde=True, ax=axes[1, 1])
axes[1, 1].set_title("(d) Distribusi BMI")

plt.tight_layout()
plt.savefig("outputs/gambar2_distribusi_klinis.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 8.2 Rasio keberhasilan per kelompok usia ---
success_by_age = df.groupby("age_group")["label_implantasi"].mean() * 100
print("\nTingkat keberhasilan (%) per kelompok usia:")
print(success_by_age)

# --- 8.3 Analisis Korelasi (Matriks Pearson 7x7) ---
corr_cols = numeric_cols + ["label_implantasi"]
corr_matrix = df[corr_cols].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap Matriks Korelasi Pearson Antar-Variabel Klinis Pasien IVF")
plt.tight_layout()
plt.savefig("outputs/gambar3_korelasi_pearson.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nKorelasi kunci:")
print("Usia vs AMH:", corr_matrix.loc["usia_ibu", "AMH"])
print("Usia vs FSH:", corr_matrix.loc["usia_ibu", "FSH"])
print("AMH vs label:", corr_matrix.loc["AMH", "label_implantasi"])
print("FSH vs label:", corr_matrix.loc["FSH", "label_implantasi"])
print("BMI vs lainnya (mean abs corr):",
      corr_matrix["BMI"].drop("BMI").abs().mean())

# ------------------------------------------------------------
# 9. SIMPAN DATA HASIL WRANGLING
# ------------------------------------------------------------
train_df.to_csv("outputs/train_processed.csv", index=False)
val_df.to_csv("outputs/val_processed.csv", index=False)
test_df.to_csv("outputs/test_processed.csv", index=False)

print("\nProses data wrangling dan EDA selesai. File hasil disimpan di folder outputs/.")
