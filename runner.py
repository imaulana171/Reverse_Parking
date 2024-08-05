import subprocess

# Daftar file yang akan dijalankan
files_to_run = ['master.py', 'main.py']

processes = []

for file in files_to_run:
    # Jalankan setiap file menggunakan subprocess.Popen
    process = subprocess.Popen(['python', file])
    processes.append(process)

# Tunggu semua proses selesai
for process in processes:
    process.wait()
