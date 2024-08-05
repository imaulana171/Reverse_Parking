import cv2
import redis
import numpy as np

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Capture frame from webcam
cam = cv2.VideoCapture('Video/3PosSalah.mp4') #(0)
while True:
    ret, frame = cam.read() #bisa rtsp
    if not ret:
        print("Failed to capture image")
        cam.release()
        exit()

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Save frame to Redis
    redis_client.set('frame', frame_bytes)

    cv2.imshow('master', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
