import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Membaca data dari file Excel
file_path = 'hasil_klasterisasi_Palang_All.xlsx'  # Gantilah dengan path sesuai lokasi file Anda
df = pd.read_excel(file_path)

# Memilih fitur dan target
features = df.iloc[:, 1:-1]  # Memilih kolom ke-3 hingga sebelum kolom terakhir
target = df['Klaster']  # Menggunakan kolom 'Klaster' sebagai target

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standarisasi fitur karena Gaussian Naive Bayes memerlukan asumsi bahwa data terdistribusi normal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat model Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Membuat prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)