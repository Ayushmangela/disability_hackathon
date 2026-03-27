import cv2
import numpy as np
import math
import time
import os
import threading
import csv
from datetime import datetime
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== PATHS & SETUP ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
FACES_DIR = os.path.join(BASE_DIR, "Data", "faces")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")
LOG_FILE = os.path.join(BASE_DIR, "gesture_log.csv")
CAPTURE_DIR = os.path.join(BASE_DIR, "Captured_Frames")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Twilio
ACCOUNT_SID = os.getenv("TWILIO_SID", "ACa8c3a6ec4e9809e86bd009471a4a4473")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "5395c627b65140ca30ef8e682f4c79ce")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER", "+12298059944")
OWNER_NUMBER = os.getenv("OWNER_NUMBER", "+918623083659")
POLICE_NUMBER = os.getenv("POLICE_NUMBER", "+918623083659")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ========================== AI MODELS ==========================
hand_detector = HandDetector(maxHands=1)
classifier = Classifier(KERAS_MODEL_PATH, LABELS_PATH)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip().split(maxsplit=1)[-1] if len(line.split()) > 1 else line.strip() for line in f.readlines()]
else:
    labels = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_id_to_name = {}

def train_face_recognizer():
    global face_id_to_name
    samples, labels_ids = [], []
    current_id = 0
    
    if not os.path.isdir(FACES_DIR):
        return

    for person_name in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        face_id_to_name[current_id] = person_name
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                samples.append(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
                labels_ids.append(current_id)
        current_id += 1
    
    if samples:
        face_recognizer.train(samples, np.array(labels_ids))

train_face_recognizer()

# ========================== SYSTEM STATE ==========================
class SystemState:
    def __init__(self):
        self.safehouse_mode = False
        self.last_face = "None"
        self.last_gesture = "None"
        self.status = "System Ready"
        self.captured_once = False
        self.unknown_start_time = 0
        self.last_trigger_time = {}

state = SystemState()
event_logs = []  # List of dicts for frontend logs
main_loop = None

def add_log(event, status):
    global event_logs
    log_entry = {
        "id": int(time.time() * 1000),
        "time": datetime.now().strftime("%H:%M:%S"),
        "event": event,
        "status": status
    }
    event_logs.insert(0, log_entry)
    event_logs = event_logs[:20]  # Keep last 20
    return log_entry

async def broadcast_state():
    await manager.broadcast({
        "type": "state_update",
        "safehouse_mode": state.safehouse_mode,
        "status": state.status,
        "last_face": state.last_face,
        "last_gesture": state.last_gesture,
        "logs": event_logs
    })

def safe_broadcast():
    if main_loop:
        import asyncio
        asyncio.run_coroutine_threadsafe(broadcast_state(), main_loop)

cap = cv2.VideoCapture(0)

# ========================== WEBSOCKETS ==========================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ========================== LOGIC & UTILS ==========================
def log_event(gesture, face_name, message=""):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), gesture, face_name, "ON" if state.safehouse_mode else "OFF", message])

async def trigger_actions(label):
    now = time.time()
    if now - state.last_trigger_time.get(label, 0) < 5: return
    state.last_trigger_time[label] = now
    
    if label == "ThumbsUp" and state.last_face == "Pratham":
        state.safehouse_mode = True
        state.status = "SafeHouse ON"
        add_log("SafeHouse Mode", "ACTIVATED")
    elif label == "ThumbsDown" and state.last_face == "Pratham":
        state.safehouse_mode = False
        state.status = "SafeHouse OFF"
        add_log("SafeHouse Mode", "DEACTIVATED")
    elif label == "Help":
        state.status = "HELP SENT"
        add_log("Emergency", "HELP ALERT SENT")
        try:
            client.messages.create(body="🚨 HELP detected!", from_=TWILIO_NUMBER, to=OWNER_NUMBER)
        except:
            pass
    
    await broadcast_state()

def trigger_actions_sync(label):
    if main_loop:
        import asyncio
        asyncio.run_coroutine_threadsafe(trigger_actions(label), main_loop)

def gen_frames():
    global cap
    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                # Show a "Camera Waiting" frame or just sleep
                black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_img, "Camera Initialize Failed", (100, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', black_img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(2)
                continue

        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
        
        img_output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        current_face = "None"
        for (x, y, w, h) in faces:
            id, conf = face_recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))
            name = face_id_to_name.get(id, "Unknown") if conf < 60 else "Unknown"
            current_face = name
            cv2.rectangle(img_output, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(img_output, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        state.last_face = current_face

        # Hand detection & Classification
        hands, _ = hand_detector.findHands(frame, draw=False)
        current_gesture = "None"
        if hands:
            x, y, w, h = hands[0]['bbox']
            cv2.rectangle(img_output, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
            
            # --- Restore Classification Logic ---
            img_size = 300
            offset = 20
            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
            y1, y2 = max(0, y - offset), min(frame.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(frame.shape[1], x + w + offset)
            img_crop = frame[y1:y2, x1:x2]

            if img_crop.size != 0:
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = img_size / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(img_crop, (wCal, img_size))
                    wGap = math.ceil((img_size - wCal) / 2)
                    img_white[:, wGap:wCal + wGap] = imgResize
                else:
                    k = img_size / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(img_crop, (img_size, hCal))
                    hGap = math.ceil((img_size - hCal) / 2)
                    img_white[hGap:hCal + hGap, :] = imgResize
                
                try:
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                    if index < len(labels):
                        current_gesture = labels[index]
                        cv2.putText(img_output, f"HAND: {current_gesture}", (x, y-40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass
        
        if current_gesture != state.last_gesture:
            state.last_gesture = current_gesture
            if current_gesture != "None":
                add_log("Gesture Detected", current_gesture)
                trigger_actions_sync(current_gesture)

        ret, buffer = cv2.imencode('.jpg', img_output)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    global main_loop
    import asyncio
    main_loop = asyncio.get_running_loop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
