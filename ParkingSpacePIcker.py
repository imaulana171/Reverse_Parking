import cv2
import pickle
 
width, height = 15, 30
 
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []
 
 
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
 
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)
 
 
while True:
    img = cv2.imread('MasjidRektorat.png') #('Screenshoot2.png')
    counter = 1
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 255, 0), 2)
        cv2.putText(img, str(counter), (pos[0]+5, pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        counter += 1
 
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)

    # Menghentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan sumber daya
cv2.destroyAllWindows()