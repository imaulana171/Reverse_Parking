import pandas as pd

# Membaca file CSV
data = pd.read_csv('Results/Pos6Salah_fps_data.csv')

# Mengambil setiap data ke-100
sampled_data = data.iloc[::10, :]

# Menyimpan ke file Excel
sampled_data.to_excel('Results/sampled_data.xlsx', index=False)
