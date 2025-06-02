from flask import Flask, render_template, request
import pickle
from datetime import datetime

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Daftar gejala
gejala = [
    "demam", "batuk", "sakit_kepala", "nyeri_otot", "sesak_nafas",
    "pilek", "mual", "muntah", "diare", "sakit_tenggorokan",
    "hilang_penciuman", "ruam_kulit", "mata_merah", "nyeri_perut", "pusing"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    nama = request.form.get("nama")
    tahun_lahir = int(request.form.get("tahun_lahir"))
    tahun_sekarang = datetime.now().year
    umur = tahun_sekarang - tahun_lahir

    
    data = []
    for g in gejala:
        data.append(1 if request.form.get(g) == "1" else 0)

    # Prediksi
    hasil = model.predict([data])[0]

    return render_template("index.html", hasil=hasil, nama=nama, umur=umur)

if __name__ == "__main__":
    app.run(debug=True)
