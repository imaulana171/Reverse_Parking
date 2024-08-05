import cv2
import redis
import numpy as np

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Load the image
image = cv2.imread('Contoh.jpg')  # Replace with the path to your JPEG image
if image is None:
    print("Failed to load image")
    exit()

# Encode image to JPEG
_, buffer = cv2.imencode('.jpg', image)
image_bytes = buffer.tobytes()

# Save image to Redis
redis_client.set('frame', image_bytes)

# Retrieve image from Redis
retrieved_image_bytes = redis_client.get('frame')
nparr = np.frombuffer(retrieved_image_bytes, np.uint8)
retrieved_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Display the image
cv2.imshow('Image', retrieved_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
