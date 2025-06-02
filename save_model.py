from sklearn.tree import DecisionTreeClassifier
import pickle

# Daftar 15 gejala
gejala = [
    "demam", "batuk", "sakit_kepala", "nyeri_otot", "sesak_nafas",
    "pilek", "mual", "muntah", "diare", "sakit_tenggorokan",
    "hilang_penciuman", "ruam_kulit", "mata_merah", "nyeri_perut", "pusing"
]

# Data pelatihan sederhana (dummy data)
X = [
    [1,1,1,1,1,0,0,0,0,1,0,0,0,0,1],  # Flu
    [1,1,1,0,0,0,1,1,0,0,0,0,0,0,1],  # DBD
    [1,1,1,1,1,0,0,0,1,1,1,0,0,0,1],  # Covid
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # Sehat
    [1,1,1,1,0,1,1,0,0,0,0,1,1,0,0],  # Campak
    [1,1,0,0,0,0,1,1,1,1,1,0,0,1,1],  # Tipes
    [0,1,1,0,1,1,0,0,0,1,1,0,0,0,1],  # ISPA
    [1,0,1,1,1,0,0,0,1,0,0,1,0,1,1],  # Malaria
    [0,0,1,1,1,0,0,0,1,1,1,0,0,0,1],  # Asma
    [1,0,0,1,0,1,0,0,1,0,0,1,1,1,0],  # Alergi
]
y = [
    "Flu", "DBD", "Covid", "Sehat", "Campak", 
    "Tipes", "ISPA", "Malaria", "Asma", "Alergi"
]

# Buat dan latih model
model = DecisionTreeClassifier()
model.fit(X, y)

# Simpan model ke file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan dengan 15 gejala dan 10 penyakit.")
