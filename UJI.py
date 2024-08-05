import cv2
import pickle
import numpy as np
import pygame
import time
import threading
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms
import matplotlib
import redis

matplotlib.use('TkAgg')  # Atau backend lain yang sesuai dengan sistem Anda

# Path ke file model serta posisi parkir
onnx_model_path = 'Model/CustomYOLOv5s.onnx'
parking_positions_path = 'CarParkPos'
audio_awalan_path = 'Sound/awalan.mp3'
audio_alarm_path_template = 'Sound/alarm ({}).mp3'

# Load model YOLO menggunakan PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.hub.load('ultralytics/yolov5', 'custom', path=onnx_model_path).to(device)
classes = ['Front', 'Rear']

# Load posisi parkir dari file pickle
try:
    with open(parking_positions_path, 'rb') as f:
        posList = pickle.load(f)
except Exception as e:
    print(f"Error loading parking positions: {e}")
    exit(1)

# Inisialisasi Pygame untuk pemutaran audio
pygame.init()

def music_busy():
    # Menunggu sampai musik selesai dimainkan
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(4)

def announce(number):
    # Fungsi untuk mengumumkan nomor slot yang terdeteksi
    global slot_numbers_annced, last_trigger_time
    slot_numbers_annced = number.copy()
    pygame.mixer.music.load(audio_awalan_path)
    pygame.mixer.music.play()
    music_busy()
    for y in set(number):
        pygame.mixer.music.load(audio_alarm_path_template.format(y))
        pygame.mixer.music.play()
        music_busy()
    last_trigger_time = time.time()

# Variabel global untuk kontrol alarm
is_alarm_playing = False
trigger_alarm = False
last_trigger_time = 0
frames = 0
slot_numbers = []
slot_numbers_annced = []

# Inisialisasi variabel perhitungan FPS
fps_list = []
start_time = time.time()

# Inisialisasi hasil deteksi
detection_results = []

# Buffer untuk konsistensi deteksi
detection_buffer = []
buffer_size = 10

def process_frame(img):
    global detection_results, detection_buffer, slot_numbers, frames
    current_time = time.time() - start_time
    # Mengubah gambar menjadi tensor PyTorch dan memindahkannya ke GPU
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Melakukan inferensi
    with torch.no_grad():
        detections = net(img_tensor)[0].cpu().numpy()

    # Pra-pemrosesan gambar
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 1)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    
    classes_ids = []
    confidences = []
    car_boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.1:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w / 2) * x_scale)
                y1 = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                car_boxes.append(box)

    frame_slot_numbers = []
    frame_detections = []
    for i, pos in enumerate(posList):
        x, y = pos
        found = False
        car_type = None
        for j, car_box in enumerate(car_boxes):
            car_x, car_y, car_w, car_h = car_box
            if car_x < x + 20 and car_x + car_w > x and car_y < y + 30 and car_y + car_h > y:
                slot_number = i + 1
                if j < len(classes_ids):  # Pastikan indeks berada dalam rentang
                    car_type = classes[classes_ids[j]]
                    confidence = confidences[j]
                    if car_type == 'Rear':
                        frame_slot_numbers.append(slot_number)
                    frame_detections.append({
                        'Frame': frames,
                        'Time (s)': current_time,
                        'Posisi': slot_number,
                        'Deteksi': 'Ada Mobil',
                        'Tipe Mobil': car_type,
                        'Confidence': confidence,
                        'Alarm Played': False,
                        'Announced Slots': ''
                    })
                    color = (0, 165, 255) if car_type == 'Rear' else (0, 255, 0)
                found = True
                break

        if not found:
            color = (0, 0, 255)
            frame_detections.append({
                'Frame': frames,
                'Time (s)': current_time,
                'Posisi': i + 1,
                'Deteksi': 'Kosong',
                'Tipe Mobil': '-',
                'Confidence': '-',
                'Alarm Played': False,
                'Announced Slots': ''
            })

        cv2.rectangle(img, pos, (pos[0] + 20, pos[1] + 30), color, 3)
        cv2.putText(img, f'{i + 1}', (pos[0] + 5, pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    detection_buffer.append(frame_slot_numbers)
    if len(detection_buffer) > buffer_size:
        detection_buffer.pop(0)
    
    # Mengambil hasil deteksi yang konsisten dari buffer
    consistent_detections = set(detection_buffer[0])
    for buffer in detection_buffer:
        consistent_detections.intersection_update(buffer)

    slot_numbers = list(consistent_detections)

    indices = cv2.dnn.NMSBoxes(car_boxes, confidences, 0.3, 0.3)
    for i in indices:
        x1, y1, w, h = car_boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + " {:.2f}".format(conf)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    detection_results.extend(frame_detections)

    return img

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Loop pemrosesan video
while True:
    frames += 1

    # Retrieve frame from Redis
    frame_bytes = redis_client.get('frame')
    if frame_bytes is None:
        print("Failed to retrieve image from Redis")
        exit()

    # Decode the byte string to a numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        break

    start_frame_time = time.time()
    img = process_frame(img)

    # Menghitung FPS dan menambahkannya ke daftar fps_list
    fps = 1 / (time.time() - start_frame_time)
    fps_list.append(fps)

    # Menampilkan FPS di layar
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Parking Space", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    try:
        lanjut = anno.is_alive()
    except:
        lanjut = False

    if (len(slot_numbers) > 0) and (not lanjut) and ((last_trigger_time == 0) or (last_trigger_time < (time.time() - 10))):
        anno = threading.Thread(target=announce, args=(slot_numbers,))
        anno.start()

        # Update detection results with alarm status and announced slots
        for result in detection_results:
            if result['Frame'] == frames and result['Posisi'] in slot_numbers:
                result['Alarm Played'] = True
                result['Announced Slots'] = ','.join(map(str, slot_numbers_annced))

cv2.destroyAllWindows()

# Plot grafik FPS
plt.figure(figsize=(10, 5))
plt.plot(fps_list, label='FPS')
plt.xlabel('Frame')
plt.ylabel('FPS')
plt.title('FPS over Time')
plt.legend()
plt.show(block=True)
plt.savefig('fps_graph.png')  # Menyimpan grafik ke file PNG

# Tampilkan informasi bahwa grafik telah disimpan
print("Grafik FPS telah disimpan sebagai 'fps_graph.png'")

# Tampilkan tabel hasil deteksi
df = pd.DataFrame(detection_results)
print(df)

# Menyimpan hasil deteksi ke CSV
df.to_csv('detection_results.csv', index=False)
print("Hasil deteksi telah disimpan sebagai 'detection_results.csv'")

# Simpan data FPS ke CSV
fps_df = pd.DataFrame(fps_list, columns=['FPS'])
fps_df.to_csv('fps_data.csv', index=False)
print("Data FPS telah disimpan sebagai 'fps_data.csv'")

# Hitung durasi pemutaran video
total_duration = time.time() - start_time
duration_df = pd.DataFrame([{'Total Duration (seconds)': total_duration}])
duration_df.to_csv('video_duration.csv', index=False)
print("Durasi pemutaran video telah disimpan sebagai 'video_duration.csv'")

# Buka file gambar grafik FPS
import os
os.startfile('fps_graph.png')
