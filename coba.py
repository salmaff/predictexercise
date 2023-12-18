import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Membaca data dari file Excel
file_path = 'Palang_All.xlsx'  # Gantilah dengan path sesuai lokasi file Anda
df = pd.read_excel(file_path)

# Memilih fitur untuk klasterisasi
features = df.iloc[:, 2:]

# Standarisasi fitur
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Menentukan jumlah klaster
num_clusters = 6

# Melakukan klasterisasi menggunakan K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Klaster'] = kmeans.fit_predict(scaled_features)

# Menyimpan hasil klasterisasi ke dalam file Excel
output_file_path = 'hasil_klasterisasi_Palang_All.xlsx'
df.to_excel(output_file_path, index=False)

print(f"Hasil klasterisasi disimpan ke dalam file: {output_file_path}")
