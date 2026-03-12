import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import math
import time
import face_recognition
import os
import threading
import csv
from datetime import datetime
from twilio.rest import Client

# ========================== PATH SETUP ==========================
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "Model", "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "Model", "labels.txt")
FACES_PATH = os.path.join(BASE_DIR, "Data", "faces")
CAPTURE_DIR = os.path.join(BASE_DIR, "Captured_Frames")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ========================== CSV LOGGING ==========================
log_file = os.path.join(BASE_DIR, "gesture_log.csv")
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gesture", "FaceName", "SafeHouseMode", "Message"])

# ========================== DETECTORS ==========================
hand_detector = HandDetector(maxHands=1)

# ========================== FACE RECOGNITION ==========================
images = []
classNames = []

if os.path.exists(FACES_PATH):
    for person_name in os.listdir(FACES_PATH):
        person_folder = os.path.join(FACES_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                images.append(curImg)
                classNames.append(person_name)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = findEncodings(images)

# ========================== CLASSIFIER ==========================
classifier = Classifier(MODEL_PATH, LABEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = [line.strip().split(maxsplit=1)[-1] for line in f.readlines()]

offset = 20
imgSize = 300

# ========================== TWILIO SETUP ==========================
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
owner_number = os.getenv("OWNER_NUMBER")
police_number = os.getenv("POLICE_NUMBER")

client = Client(account_sid, auth_token) if account_sid and auth_token else None

# ========================== STATE VARIABLES ==========================
last_label = None
last_face_label = None
last_face_location = None
gesture_start_time = 0
triggered = False
hold_duration = 3
frame_count = 0
status_text = ""
status_expire = 0
cooldown_seconds = 5
last_trigger_time = {"Help":0, "Call":0, "Danger":0, "ThumbsUp":0, "ThumbsDown":0}
safehouse_mode = False
unknown_start_time = 0
unknown_hold_duration = 3
captured_once = False

# ========================== UTILITIES ==========================
def log_event(gesture, face_name, message=""):
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            gesture,
            face_name if face_name else "Unknown",
            "ON" if safehouse_mode else "OFF",
            message
        ])

def set_status(text, duration=3):
    global status_text, status_expire
    status_text = text
    status_expire = time.time() + duration

def async_send_sms(to, message):
    if not client:
        return
    def job():
        try:
            client.messages.create(body=message, from_=twilio_number, to=to)
        except:
            pass
    threading.Thread(target=job, daemon=True).start()

def async_make_call(to, message):
    if not client:
        return
    def job():
        try:
            client.calls.create(
                twiml=f'<Response><Say>{message}</Say></Response>',
                from_=twilio_number,
                to=to
            )
        except:
            pass
    threading.Thread(target=job, daemon=True).start()

# ========================== ACTION TRIGGER ==========================
def trigger_actions(label):
    global safehouse_mode, unknown_start_time, last_face_label

    gesture_name = label.replace(" ", "").strip()
    now = time.time()

    if gesture_name not in last_trigger_time:
        return

    if now - last_trigger_time[gesture_name] < cooldown_seconds:
        return

    last_trigger_time[gesture_name] = now

    if gesture_name == "ThumbsUp" and last_face_label == "Pratham":
        safehouse_mode = True
        unknown_start_time = 0
        log_event("ThumbsUp", last_face_label, "SafeHouse Mode ON")

    elif gesture_name == "ThumbsDown" and last_face_label == "Pratham":
        safehouse_mode = False
        log_event("ThumbsDown", last_face_label, "SafeHouse Mode OFF")

    elif gesture_name == "Help":
        async_send_sms(owner_number, "HELP detected!")
        log_event("Help", last_face_label)

    elif gesture_name == "Call":
        async_make_call(owner_number, "Emergency call request.")
        log_event("Call", last_face_label)

    elif gesture_name == "Danger":
        async_send_sms(owner_number, "Danger detected!")
        async_send_sms(police_number, "Possible threat detected.")
        log_event("Danger", last_face_label)

# ========================== MAIN FRAME PROCESSOR ==========================
def process_frame(img):
    global frame_count, last_face_label, last_face_location
    global last_label, gesture_start_time, triggered
    global unknown_start_time, captured_once, safehouse_mode
    global status_text, status_expire

    frame_count += 1
    imgOutput = img.copy()
    frame_h, frame_w = img.shape[:2]

    hands, img = hand_detector.findHands(img, flipType=False)
    detected_region = None
    label = None

    # -------- HAND DETECTION --------
    if hands:
        x, y, w, h = hands[0]['bbox']
        detected_region = (x, y, w, h)

    # -------- FACE RECOGNITION (every 5 frames) --------
    if frame_count % 5 == 0 and len(encodeListKnown) > 0:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        if facesCurFrame:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    last_face_label = classNames[matchIndex]
                else:
                    last_face_label = "Unknown"

                last_face_location = faceLoc
        else:
            last_face_label = None
            last_face_location = None

    # -------- GESTURE CLASSIFICATION --------
    if detected_region:
        x, y, w, h = detected_region
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        y1, y2 = max(0, y-offset), min(frame_h, y+h+offset)
        x1, x2 = max(0, x-offset), min(frame_w, x+w+offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            try:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if index < len(labels):
                    label = labels[index]
            except:
                pass

    # -------- HOLD LOGIC --------
    if label:
        current_time = time.time()
        if label != last_label:
            last_label = label
            gesture_start_time = current_time
            triggered = False

        if current_time - gesture_start_time >= hold_duration and not triggered:
            triggered = True
            trigger_actions(label)

    else:
        last_label = None
        triggered = False

    if status_text and time.time() >= status_expire:
        status_text = ""
    
    return {
        "gesture": label,
        "face": last_face_label,
        "safehouse": safehouse_mode,
        "status": status_text
    }