from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Daftar gejala (pastikan urutannya sesuai dengan pelatihan)
gejala_list = [
    "demam", "batuk", "sakit_kepala", "nyeri_otot", "sesak_nafas",
    "pilek", "mual", "muntah", "diare", "sakit_tenggorokan",
    "hilang_penciuman", "ruam_kulit", "mata_merah", "nyeri_perut", "pusing",
    "kehilangan_nafsu_makan", "kedinginan", "detak_jantung_cepat", "keringat_dingin", "menggigil"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    nama = request.form.get("nama", "Pengguna")
    
    # Ambil input gejala sebagai angka 0/1
    data_input = [1 if request.form.get(gj) == "1" else 0 for gj in gejala_list]

    # Prediksi penyakit
    hasil_prediksi = model.predict([data_input])[0]

    return render_template("index.html", hasil=hasil_prediksi, nama=nama)

if __name__ == "__main__":
    app.run(debug=True)
