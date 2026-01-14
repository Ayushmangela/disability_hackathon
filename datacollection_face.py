import cv2
import time
import os

cap = cv2.VideoCapture(0)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

folder = r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Data\faces\Mheet"
os.makedirs(folder, exist_ok=True)
counter = 0
imgSize = 300 

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        imgCrop = img[y:y + h, x:x + w]
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        cv2.imshow("FaceCrop", imgResize)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Face_{time.time()}.jpg', imgResize)
        print(f"Saved: {counter} images")

    elif key == ord("q"):  
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
