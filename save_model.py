from sklearn.tree import DecisionTreeClassifier
import pickle

# Daftar gejala
gejala = [
    "demam", "batuk", "sakit_kepala", "nyeri_otot", "sesak_nafas",
    "pilek", "mual", "muntah", "diare", "sakit_tenggorokan",
    "hilang_penciuman", "ruam_kulit", "mata_merah", "nyeri_perut", "pusing",
    "kehilangan_nafsu_makan", "kedinginan", "detak_jantung_cepat", "keringat_dingin", "menggigil"
]

# Data pelatihan sederhana (dummy, bisa kamu kembangkan sendiri)
X = [
    [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Flu
    [1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],  # DBD
    [1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],  # Covid
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Sehat
    [1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0],  # Campak
    [1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],  # Tipes
    [0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0],  # ISPA
    [1,0,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1],  # Malaria
    [0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0],  # Asma
    [1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,0,1,0,0],  # Alergi
]
y = [
    "Flu", "DBD", "Covid", "Sehat", "Campak", 
    "Tipes", "ISPA", "Malaria", "Asma", "Alergi"
]

# Buat model dan latih
model = DecisionTreeClassifier()
model.fit(X, y)

# Simpan model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan dengan 20 gejala.")
