"""
Main_app.py — Importable detection module for SafeHomeCam
Refactored from Main.py so app.py can call process_frame() without
running a standalone camera loop.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import threading
import csv
import os
from datetime import datetime
from playsound3 import playsound
from twilio.rest import Client
import face_recognition
import json
from dotenv import load_dotenv

# ========================== PATHS (relative to this file) ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_PATH = os.path.join(BASE_DIR, "Data", "faces")
MODEL_H5 = os.path.join(BASE_DIR, "Model", "keras_model.h5")
MODEL_LABELS = os.path.join(BASE_DIR, "Model", "labels.txt")
LOG_FILE = os.path.join(BASE_DIR, "gesture_log.csv")
CAPTURE_DIR = os.path.join(BASE_DIR, "Captured_Frames")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
DANGER_PATH = os.path.join(BASE_DIR, "Danger.wav")
SIREN_PATH = os.path.join(BASE_DIR, "siren.mp3")

# ========================== CSV LOG ==========================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gesture", "FaceName", "SafeHouseMode", "Message"])

# ========================== HAND DETECTOR ==========================
hand_detector = HandDetector(maxHands=1)

# ========================== FACE RECOGNITION ==========================
images = []
classNames = []
if not os.path.exists(FACES_PATH):
    os.makedirs(FACES_PATH)

for person_name in os.listdir(FACES_PATH):
    person_folder = os.path.join(FACES_PATH, person_name)
    if not os.path.isdir(person_folder):
        continue
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        curImg = cv2.imread(img_path)
        if curImg is None:
            continue
        images.append(curImg)
        classNames.append(person_name)

def _find_encodings(imgs):
    encode_list = []
    for img in imgs:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

encodeListKnown = _find_encodings(images)
print(f"[Main_app] Loaded {len(encodeListKnown)} face encodings from {FACES_PATH}")

# ========================== GESTURE CLASSIFIER ==========================
classifier = Classifier(MODEL_H5, MODEL_LABELS)

with open(MODEL_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]
labels = [l.split(maxsplit=1)[-1] if len(l.split()) > 1 else l for l in labels]
print(f"[Main_app] Gesture labels: {labels}")

offset = 20
imgSize = 300

# ========================== TWILIO SETUP ==========================
load_dotenv()
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(account_sid, auth_token)
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
owner_number = "+918623083659"
police_number = "+918623083659"
caretaker_number = "+918623083659"  # CareTaker number for voice requests

# ========================== STATE VARIABLES ==========================
last_label = None
last_face_label = None
last_face_location = None
gesture_start_time = 0
triggered = False
hold_duration = 3
frame_count = 0
status_text = ""
activation_word = "blue mango"
authorized_user = "Pratham"

def reload_settings():
    global activation_word, authorized_user
    SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                activation_word = data.get("activation_word", "blue mango").lower().strip()
                authorized_user = data.get("authorized_user", "Pratham").lower().strip()
                print(f"[Main_app] Activation word: '{activation_word}', Authorized User: '{authorized_user}'")
        except Exception as e:
            print(f"[Main_app] Error loading settings: {e}")

reload_settings()

status_expire = 0
cooldown_seconds = 5
last_trigger_time = {"Help": 0, "Call": 0, "Danger": 0, "ThumbsUp": 0, "ThumbsDown": 0}

safehouse_mode = False
unknown_start_time = 0
unknown_hold_duration = 3
captured_once = False

# ========================== SAFEVOICEMODE STATE ==========================
safe_voice_mode = False
last_voice_command = None
voice_command_time = 0

# ========================== UTILITY FUNCTIONS ==========================
def log_event(gesture, face_name, message=""):
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                gesture,
                face_name if face_name else "Unknown",
                "ON" if safehouse_mode else "OFF",
                message
            ])
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def safe_play(path):
    if os.path.isfile(path):
        try:
            playsound(path)
        except Exception as e:
            print(f"[Sound Error] {e}")


def set_status(text, duration=3):
    global status_text, status_expire
    status_text = text
    status_expire = time.time() + duration


def send_sms_sync(to, message):
    try:
        msg = twilio_client.messages.create(body=message, from_=twilio_number, to=to)
        return ("ok", getattr(msg, "sid", None))
    except Exception as e:
        return ("error", str(e))


def make_call_sync(to, message):
    try:
        call = twilio_client.calls.create(
            twiml=f'<Response><Say>{message}</Say></Response>',
            from_=twilio_number, to=to
        )
        return ("ok", getattr(call, "sid", None))
    except Exception as e:
        return ("error", str(e))


def async_send_sms(to, message):
    def job():
        res = send_sms_sync(to, message)
        set_status(f"SMS -> {to}: {res[0]}", 4)
    threading.Thread(target=job, daemon=True).start()


def async_make_call(to, message):
    def job():
        res = make_call_sync(to, message)
        set_status(f"Call -> {to}: {res[0]}", 4)
    threading.Thread(target=job, daemon=True).start()


# ========================== TRIGGER ACTIONS ==========================
def trigger_actions(label):
    global safehouse_mode, unknown_start_time, last_face_label

    gesture_name = label.replace(" ", "").strip()

    if gesture_name not in ["Help", "Call", "Danger", "ThumbsUp", "ThumbsDown"]:
        return

    now = time.time()
    if now - last_trigger_time.get(gesture_name, 0) < cooldown_seconds:
        set_status(f"{gesture_name} (cooldown)", 2)
        return
    last_trigger_time[gesture_name] = now

    if gesture_name == "ThumbsUp":
        if last_face_label.lower() == authorized_user.lower():
            safehouse_mode = True
            unknown_start_time = 0
            set_status(f"SafeHouse Mode ON (Authorized: {authorized_user})", 5)
            log_event("ThumbsUp", last_face_label, "SafeHouse Mode turned ON")
        else:
            set_status(f"Access Denied: Only {authorized_user} can turn ON SafeHouse Mode", 5)
            log_event("ThumbsUp", last_face_label, "Access Denied")

    elif gesture_name == "ThumbsDown":
        if last_face_label.lower() == authorized_user.lower():
            safehouse_mode = False
            set_status(f"SafeHouse Mode OFF (Authorized: {authorized_user})", 5)
            unknown_start_time = 0
            log_event("ThumbsDown", last_face_label, "SafeHouse Mode turned OFF")
        else:
            set_status(f"Access Denied: Only {authorized_user} can turn OFF SafeHouse Mode", 5)
            log_event("ThumbsDown", last_face_label, "Access Denied")

    elif gesture_name == "Help":
        set_status("HELP triggered", 5)
        threading.Thread(target=safe_play, args=(DANGER_PATH,), daemon=True).start()
        async_send_sms(owner_number, "\U0001f6a8 HELP detected! Immediate assistance may be required.")
        log_event("Help", last_face_label, "HELP gesture triggered")

    elif gesture_name == "Call":
        set_status("CALL triggered", 5)
        async_make_call(owner_number, "Emergency call request received. Please check immediately.")
        async_send_sms(owner_number, "\U0001f4de CALL gesture detected. Call initiated.")
        log_event("Call", last_face_label, "CALL gesture triggered")

    elif gesture_name == "Danger":
        set_status("DANGER triggered", 6)
        threading.Thread(target=safe_play, args=(DANGER_PATH,), daemon=True).start()
        async_send_sms(owner_number, "\u26a0\ufe0f DANGER ALERT! Something unusual detected.")
        async_send_sms(police_number, "\U0001f6a8 Possible threat detected at the registered address.")
        async_make_call(owner_number, "Danger alert triggered! Authorities have been notified.")
        log_event("Danger", last_face_label, "DANGER gesture triggered")


# ========================== VOICE COMMAND HANDLER ==========================
def handle_voice_command(command):
    """
    Handle a voice command from the SafeVoiceMode feature.
    Returns dict with result info.
    """
    global safe_voice_mode, last_voice_command, voice_command_time
    global safehouse_mode, unknown_start_time

    cmd = command.lower().strip()
    last_voice_command = cmd
    voice_command_time = time.time()

    # Activation / deactivation phrase
    if activation_word in cmd:
        if last_face_label.lower() == authorized_user.lower():
            safe_voice_mode = not safe_voice_mode
            state = "ON" if safe_voice_mode else "OFF"
            set_status(f"SafeVoiceMode {state} (Auth: {authorized_user})", 5)
            log_event("VoiceCommand", last_face_label, f"SafeVoiceMode turned {state}")
            return {
                "action": "toggle_safevoice",
                "safe_voice_mode": safe_voice_mode,
                "message": f"SafeVoiceMode {state} (Authorized)"
            }
        else:
            set_status(f"Wake-word Denied: Recognized as {last_face_label}", 5)
            log_event("WakeWordDenied", last_face_label, "Unauthorized wake-word attempt")
            return {
                "action": "ignored",
                "safe_voice_mode": safe_voice_mode,
                "message": f"Access Denied: Only {authorized_user} can activate SafeVoiceMode"
            }

    # Only process action commands when SafeVoiceMode is active
    if not safe_voice_mode:
        return {
            "action": "ignored",
            "safe_voice_mode": False,
            "message": f"SafeVoiceMode is OFF. Say '{activation_word}' to activate."
        }

    # CRITICAL: RESTRICT ALL ACTION COMMANDS TO AUTHORIZED USER
    if last_face_label.lower() != authorized_user.lower():
        set_status(f"Action Denied: {last_face_label} not authorized", 5)
        log_event("VoiceActionDenied", last_face_label, f"Attempted command: {cmd}")
        return {
            "action": "denied",
            "safe_voice_mode": True,
            "message": f"Access Denied: Only {authorized_user} can issue commands."
        }

    # ---- SafeHouse toggle via voice ----
    if "thumbs up" in cmd or "thumbsup" in cmd:
        safehouse_mode = True
        unknown_start_time = 0
        set_status(f"VOICE: SafeHouse Mode ON (Auth: {authorized_user})", 5)
        log_event("VoiceThumbsUp", last_face_label, "SafeHouse ON via voice")
        return {
            "action": "thumbsup",
            "safe_voice_mode": True,
            "message": f"SafeHouse Mode turned ON (Welcome {authorized_user})"
        }

    elif "thumbs down" in cmd or "thumbsdown" in cmd:
        safehouse_mode = False
        unknown_start_time = 0
        set_status(f"VOICE: SafeHouse Mode OFF (Auth: {authorized_user})", 5)
        log_event("VoiceThumbsDown", last_face_label, "SafeHouse OFF via voice")
        return {
            "action": "thumbsdown",
            "safe_voice_mode": True,
            "message": f"SafeHouse Mode turned OFF (Goodbye {authorized_user})"
        }

    # ---- CareTaker requests via voice ----
    elif "water" in cmd:
        set_status("VOICE: Water request sent", 5)
        async_send_sms(caretaker_number, "\U0001f4a7 Voice Request: Please bring water. The person needs hydration.")
        log_event("VoiceWater", last_face_label, "Water request sent to CareTaker")
        return {
            "action": "water",
            "safe_voice_mode": True,
            "message": "Water request sent to CareTaker"
        }

    elif "hungry" in cmd or "food" in cmd:
        set_status("VOICE: Food request sent", 5)
        async_send_sms(caretaker_number, "\U0001f35d Voice Request: The person is hungry. Please bring food.")
        log_event("VoiceHungry", last_face_label, "Food request sent to CareTaker")
        return {
            "action": "hungry",
            "safe_voice_mode": True,
            "message": "Food request sent to CareTaker"
        }

    elif "medicine" in cmd or "medicines" in cmd:
        set_status("VOICE: Medicine request sent", 5)
        async_send_sms(caretaker_number, "\U0001f48a Voice Request: Please bring medicines. The person needs medication.")
        log_event("VoiceMedicine", last_face_label, "Medicine request sent to CareTaker")
        return {
            "action": "medicine",
            "safe_voice_mode": True,
            "message": "Medicine request sent to CareTaker"
        }

    # ---- Emergency commands ----
    elif "help" in cmd:
        set_status("VOICE: HELP triggered", 5)
        threading.Thread(target=safe_play, args=(DANGER_PATH,), daemon=True).start()
        async_send_sms(owner_number, "\U0001f6a8 VOICE HELP! Immediate assistance required.")
        log_event("VoiceHelp", last_face_label, "Voice command HELP triggered")
        return {
            "action": "help",
            "safe_voice_mode": True,
            "message": "HELP action triggered via voice"
        }

    elif "call" in cmd:
        set_status("VOICE: CALL triggered", 5)
        async_make_call(owner_number, "Emergency voice call request. Please check immediately.")
        async_send_sms(owner_number, "\U0001f4de VOICE CALL detected. Call initiated.")
        log_event("VoiceCall", last_face_label, "Voice command CALL triggered")
        return {
            "action": "call",
            "safe_voice_mode": True,
            "message": "CALL action triggered via voice"
        }

    elif "danger" in cmd:
        set_status("VOICE: DANGER triggered", 6)
        threading.Thread(target=safe_play, args=(DANGER_PATH,), daemon=True).start()
        async_send_sms(owner_number, "\u26a0\ufe0f VOICE DANGER ALERT! Something unusual detected.")
        async_send_sms(police_number, "\U0001f6a8 Voice danger alert at the registered address.")
        async_make_call(owner_number, "Voice danger alert! Authorities have been notified.")
        log_event("VoiceDanger", last_face_label, "Voice command DANGER triggered")
        return {
            "action": "danger",
            "safe_voice_mode": True,
            "message": "DANGER action triggered via voice"
        }

    return {
        "action": "unknown",
        "safe_voice_mode": True,
        "message": f"Unknown command: '{cmd}'"
    }


# ========================== PROCESS FRAME ==========================
def process_frame(frame):
    """
    Process a single camera frame:
      - Hand detection + gesture classification
      - Face recognition (every 5th call)
      - Gesture hold timer + trigger logic
      - SafeHouse unknown person detection
    Returns dict: {gesture, face, safehouse, status}
    """
    global last_label, last_face_label, last_face_location
    global gesture_start_time, triggered, frame_count
    global safehouse_mode, unknown_start_time, captured_once
    global status_text, status_expire

    frame_h, frame_w = frame.shape[:2]
    img = frame.copy()

    # ---------- Hand detection ----------
    hands, img = hand_detector.findHands(img, flipType=False)
    detected_region = None
    region_type = None
    label = None

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        detected_region = (x, y, w, h)
        region_type = "hand"

    # ---------- Face recognition (every 5th call) ----------
    frame_count += 1
    if frame_count % 5 == 0:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        if facesCurFrame and encodeListKnown:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    last_face_label = name
                    last_face_location = faceLoc
                else:
                    last_face_label = "Unknown"
                    last_face_location = faceLoc
        else:
            last_face_label = None
            last_face_location = None

    # ---------- SafeHouse unknown detection ----------
    if safehouse_mode:
        if last_face_label == "Unknown":
            if unknown_start_time == 0:
                unknown_start_time = time.time()
            elif time.time() - unknown_start_time >= unknown_hold_duration:
                if not captured_once:
                    set_status("UNKNOWN detected during SafeHouse Mode!", 6)
                    threading.Thread(target=safe_play, args=(SIREN_PATH,), daemon=True).start()

                    os.makedirs(CAPTURE_DIR, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    for i in range(5):
                        filename = os.path.join(CAPTURE_DIR, f"Unknown_{timestamp}_{i+1}.jpg")
                        cv2.imwrite(filename, frame)
                        time.sleep(0.3)

                    threading.Thread(target=safe_play, args=(DANGER_PATH,), daemon=True).start()
                    async_send_sms(owner_number, "Unknown person detected during SafeHouse Mode! 5 images captured.")
                    async_send_sms(police_number, "Possible intrusion detected at SafeHouse.")
                    async_make_call(owner_number, "Unknown person detected during SafeHouse Mode. Authorities have been notified.")
                    captured_once = True
                    log_event("UnknownFace", "Unknown", "Unknown person detected during SafeHouse Mode")
        else:
            unknown_start_time = 0
            captured_once = False

    # ---------- Hand gesture classification ----------
    if detected_region and region_type == "hand":
        x, y, w, h = detected_region
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        y1, y2 = max(0, y - offset), min(frame_h, y + h + offset)
        x1, x2 = max(0, x - offset), min(frame_w, x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            # Grayscale
            gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            # Contrast stretching
            min_val, max_val = np.min(gray), np.max(gray)
            if max_val - min_val > 0:
                gray = ((gray - min_val) / (max_val - min_val)) * 255
            gray = np.uint8(gray)
            # Gamma correction
            gamma = 1.5
            gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            imgEnhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            aspectRatio = h / w
            try:
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
            except Exception:
                pass

            try:
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if index < len(labels):
                    label = labels[index]
            except Exception:
                pass

    # ---------- Gesture hold timer ----------
    if label and region_type == "hand":
        current_time = time.time()
        if label != last_label:
            last_label = label
            gesture_start_time = current_time
            triggered = False

        elapsed = current_time - gesture_start_time
        if elapsed >= hold_duration and not triggered:
            triggered = True
            set_status(f"{label} gesture triggered after 3s", 4)
            trigger_actions(label)
    else:
        last_label = None
        gesture_start_time = 0
        triggered = False

    # ---------- Clear expired status ----------
    if status_text and time.time() >= status_expire:
        status_text = ""

    # ---------- Return result ----------
    return {
        "gesture": label,
        "face": last_face_label,
        "safehouse": safehouse_mode,
        "status": status_text if time.time() < status_expire else "",
    }
# ========================== MODEL WARMUP ==========================
def warmup_models():
    """Run dummy inferences to 'warm up' the AI models at startup."""
    print("[Main_app] Warming up AI models...")
    try:
        # 1. Warm up Gesture Classifier
        dummy_img = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        classifier.getPrediction(dummy_img, draw=False)
        
        # 2. Warm up Face Recognition (if encodings loaded)
        if encodeListKnown:
            dummy_face = np.zeros((100, 100, 3), np.uint8)
            face_recognition.face_encodings(dummy_face)
            
        print("[Main_app] Warmup complete.")
    except Exception as e:
        print(f"[Main_app] Warmup error: {e}")

# Start warmup in background
threading.Thread(target=warmup_models, daemon=True).start()
