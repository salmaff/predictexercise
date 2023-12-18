from flask import Flask, render_template, request
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Membaca data dari file Excel
file_path = 'hasil_klasterisasi_Palang_All.xlsx'
df = pd.read_excel(file_path)

# Memilih fitur
features = df.iloc[:, 2:-2]

# Standarisasi fitur karena Gaussian Naive Bayes memerlukan asumsi bahwa data terdistribusi normal
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Membuat model Gaussian Naive Bayes
target = df['Klaster']
model = GaussianNB()
model.fit(features_scaled, target)

# Kamus untuk memetakan angka klaster ke nama klaster
cluster_mapping = {
    0: 'Total nilai Anda rendah denga nilai variatif di setiap subtes',
    1: 'Total nilai Anda cenderung tinggi dengan nilai Praktek Komputer tinggi. Anda terkualifikasi sebagai calon perangkat desa',
    2: 'Nilai Anda merata',
    3: 'Nilai Anda variatif dan cenderung rendah',
    4: 'Total nilai Anda rendah. Anda tidak terkualifikasi sebagai calon perangkat desa',
    5: 'Nilai Anda variatif dan cenderung tinggi'
}

# Fungsi untuk melakukan prediksi
def predict(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return cluster_mapping[prediction[0]]

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk hasil prediksi
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        input_features = [float(request.form[f'feature{i}']) for i in range(1, 7)]
        result = predict(input_features)
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
