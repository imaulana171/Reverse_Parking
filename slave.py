import cv2
import redis
import numpy as np
import time

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Retrieve frame from Redis
while True:
    frame_bytes = redis_client.get('frame')
    if frame_bytes is None:
        print("Failed to retrieve image from Redis")
        exit()

    # Decode the byte string to a numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Display the image
    time.sleep(0.1)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
