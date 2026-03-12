import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

BASE_DIR = os.getcwd()
folder = os.path.join(BASE_DIR, "Data", "Call")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # clamp crop region to frame bounds to avoid empty slices
        img_h, img_w = img.shape[:2]
        y1 = max(0, y - offset)
        y2 = min(img_h, y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img_w, x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            cv2.imshow("Image", img)
            continue

        aspectRatio = h / w if w != 0 else 1

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap : wCal + wGap] = imgResize

        else:
            k = imgSize / w if w != 0 else 1
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap : hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)