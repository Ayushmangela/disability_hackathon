import cv2 # openCV
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from playsound import playsound    
from twilio.rest import Client   
import face_recognition
import os
import threading

# ========================== PATHS, CAMERA & DETECTORS ==========================
BASE_DIR = os.getcwd()
FACES_ROOT = os.path.join(BASE_DIR, "Data", "faces")  # sub folder for each person

cap = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)

# ========================== FACE RECOGNITION ==========================
faces_path = FACES_ROOT
images = []
classNames = []
if not os.path.exists(faces_path):  #Checks if the face data folder exists. if not then creates it 
    os.makedirs(faces_path) 

# Load faces from subfolders (Pratham, Mheet, etc.)
for person_name in os.listdir(faces_path):
    person_folder = os.path.join(faces_path, person_name)
    if not os.path.isdir(person_folder):
        continue
#Loops over everything inside the faces_path directory.
# If an item isn’t a folder (e.g., a random file), skip it.
# So only person folders are processed.
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        curImg = cv2.imread(img_path)
        if curImg is None:
            continue
# Loops through each image inside that person’s folder.
# Reads it using OpenCV.
# Skips files that fail to load (e.g., corrupt or non-image files).
        images.append(curImg)
        classNames.append(person_name)
# Adds the image and its person’s name to the lists.
# These are later used for encoding and recognition.


# Converts image from BGR to RGB (OpenCV loads in BGR; face_recognition expects RGB).
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img) # converted into floating number
        if encodings:
            encodeList.append(encodings[0]) # make sure if their is only one face in the image
    return encodeList

encodeListKnown = findEncodings(images) # find encoding for the given image face

# ========================== GESTURE CLASSIFIER ==========================
classifier = Classifier(
    r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\keras_model.h5",
    r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\labels.txt"
)

with open(r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

labels = [l.split(maxsplit=1)[-1] if len(l.split()) > 1 else l for l in labels]  # their is number + name then split

offset = 20 # padding margin
imgSize = 300 # target size 

# ========================== TWILIO SETUP ==========================
account_sid = "AC7798998dc09096e2fbd6c57e20127ff4"
auth_token = "665b5bf18dd144876fa29393a1add0be"
client = Client(account_sid, auth_token)
twilio_number = "+12294588371"
owner_number = "+918623083659"
police_number = "+918623083659"  # "+919892207022"

# ========================== ALERT & SAFEHOUSE SETTINGS ==========================
last_label = None           # Last gesture detected.
last_face_label = None      # Last recognized face.
last_face_location = None    # (x, y, w, h) of last face detected. 
gesture_start_time = 0      # When a gesture started being held.
triggered = False           # Whether an alert has been triggered.
hold_duration = 3           # How long (in seconds) a gesture must be held to count as valid.
frame_count = 0             # Keeps track of frames processed.
process_every_n_frames = 5   # Only do face recognition every 5 frames to save CPU time.
status_text = ""             # Used for showing temporary status messages on the display (like “HELP SENT”).
status_expire = 0            # status_expire holds the time when the message should disappear.
cooldown_seconds = 5         # Stores the last time each gesture triggered.
                             # Enforces a cooldown between triggers.
last_trigger_time = {"Help":0, "Call":0, "Danger":0, "ThumbsUp":0, "ThumbsDown":0}

alarm_path = os.path.join(os.getcwd(), "alarm.mp3")
danger_path = os.path.join(os.getcwd(), "Danger.wav")
siren_path = os.path.join(os.getcwd(), "siren.mp3")

safehouse_mode = False #When enabled, system is on high alert (e.g., detects intruders).
unknown_start_time = 0 # Tracks when an unrecognized face appeared.
unknown_hold_duration = 3  # If the unknown face persists for >3 seconds, trigger an alert.

# ========================== UTILITY FUNCTIONS ==========================
def safe_play(path):
    if os.path.isfile(path): # Checks whether the given file (like "alarm.mp3") actually exists on disk.
        try:
            from playsound import playsound
            playsound(path)
        except:   # Runs the sound in a safe try-except block — so even if playback fails (e.g., sound driver issue), the program doesn’t crash.
            pass

def set_status(text, duration=3):
    global status_text, status_expire
    status_text = text
    status_expire = time.time() + duration

def send_sms_sync(to, message):
    try:
        msg = client.messages.create(body=message, from_=twilio_number, to=to)
        return ("ok", getattr(msg, "sid", None))
    except Exception as e:
        return ("error", str(e))

def make_call_sync(to, message):
    try:
        call = client.calls.create(twiml=f'<Response><Say>{message}</Say></Response>', from_=twilio_number, to=to)
        return ("ok", getattr(call, "sid", None))
    except Exception as e:
        return ("error", str(e))

def async_send_sms(to, message):
    def job():
        res = send_sms_sync(to, message)
        set_status(f"SMS -> {to}: {res[0]}", 4) # number : ok/error --> "SMS --> {to} : {resr[0]}"
    threading.Thread(target=job, daemon=True).start()  # daemon=True → auto-stops when program ends

def async_make_call(to, message):
    def job():
        res = make_call_sync(to, message)
        set_status(f"Call -> {to}: {res[0]}", 4)
    threading.Thread(target=job, daemon=True).start()

# ========================== TRIGGER ACTIONS ==========================
def trigger_actions(label): 
    global safehouse_mode, unknown_start_time, last_face_label # the variables can define 

    gesture_name = label.replace(" ", "").strip() # Removes spaces (e.g., "Thumbs Up" → "ThumbsUp")

    if gesture_name not in ["Help", "Call", "Danger", "ThumbsUp", "ThumbsDown"]:
        return

    now = time.time()
    if now - last_trigger_time.get(gesture_name, 0) < cooldown_seconds:
        set_status(f"{gesture_name} (cooldown)", 2) # same gesture can use after cooldown for 5s 
        return
    last_trigger_time[gesture_name] = now

    # ============================ SAFEHOUSE CONTROL (Only Pratham) ============================
    if gesture_name == "ThumbsUp":
        if last_face_label == "Pratham": # if last_face_label in ["Pratham", "Mheet"]:
            safehouse_mode = True
            unknown_start_time = 0
            set_status("SafeHouse Mode ON (Authorized: Pratham)", 5)
        else:
            set_status("Access Denied: Only Pratham can turn ON SafeHouse Mode", 5)

    elif gesture_name == "ThumbsDown":
        if last_face_label == "Pratham":
            safehouse_mode = False
            set_status("SafeHouse Mode OFF (Authorized: Pratham)", 5)
            unknown_start_time = 0
        else:
            set_status("Access Denied: Only Pratham can turn OFF SafeHouse Mode", 5)

    elif gesture_name == "Help":
        set_status("HELP triggered", 5)
        threading.Thread(target=safe_play, args=(alarm_path,), daemon=True).start()
        async_send_sms(owner_number, "🚨 HELP detected! Immediate assistance may be required.")

    elif gesture_name == "Call":
        set_status("CALL triggered", 5)
        async_make_call(owner_number, "Emergency call request received. Please check immediately.")
        async_send_sms(owner_number, "📞 CALL gesture detected. Call initiated.")

    elif gesture_name == "Danger":
        set_status("DANGER triggered", 6)
        threading.Thread(target=safe_play, args=(danger_path,), daemon=True).start()
        async_send_sms(owner_number, "⚠️ DANGER ALERT! Something unusual detected.")
        async_send_sms(police_number, "🚨 Possible threat detected at the registered address.")
        async_make_call(owner_number, "Danger alert triggered! Authorities have been notified.")

# ========================== MAIN LOOP ==========================
while True:
    success, img = cap.read() # Continuously reads frames from camera (cap).
    if not success:
        break                 # If frame not captured → exit loop.
    imgOutput = img.copy()    # Makes a duplicate image to draw rectangles, labels, etc.
    frame_h, frame_w = img.shape[:2]  # Gets frame height & width.

    hands, img = hand_detector.findHands(img, flipType=False)   # Uses cvzone.HandTrackingModule to detect hand position and bounding box.
                                                                # flipType=False → doesn’t mirror image (important for camera orientation).
    detected_region = None                                       
    region_type = None
    label = None
    frame_count += 1 

    # ----- PRIORITIZE HAND if visible -----
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  
        detected_region = (x, y, w, h)
        region_type = "hand"

            # ----- If no hand, do face recognition (every 5th frame only) -----
# ----- Run face recognition every 5th frame (always, even with hands)
    frame_count += 1  # frame_count keeps track of how many frames the camera has processed.
    if frame_count % 5 == 0:  # frame_count % 5 == 0 means: run this block once every 5 frames.
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Shrinks the image to 25% of its original size.
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # So this line swaps the color order.
        facesCurFrame = face_recognition.face_locations(imgS) # Returns a list of face bounding boxes found in the frame.
                                                            # Each box is in (top, right, bottom, left) format.
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame) # Converts detected faces into 128-dimension feature vectors (like face fingerprints).

        if facesCurFrame:
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # zip() pairs the encoding (encodeFace) with its corresponding location (faceLoc).
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #Compares the current face (encodeFace) to all known faces (encodeListKnown).
                                                                                      # gives a True/False answer (based on a threshold)
                                                                                      # Returns a list of True/False — one for each known face.
                                                                                      # [False, True, False]
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) # Calculates Euclidean distance between encodings.
                                                                                      # gives the actual distance (float value), which is more precise
                                                                                      # Smaller = more similar face.
                                                                                      # Example: [0.42, 0.25, 0.66]
                matchIndex = np.argmin(faceDis) # Returns the index of lowest distance (closest face match).

                if matches[matchIndex]:  # if true then go inside or else it is unknown
                    name = classNames[matchIndex]
                    last_face_label = name
                    last_face_location = faceLoc
                else:
                    last_face_label = "Unknown"
                    last_face_location = faceLoc

    # ---- Draw last known face box every frame ----
    if last_face_location is not None:
        y1, x2, y2, x1 = last_face_location # Remember, we resized the image to 25%.
                                            # So to draw correctly on the full-size frame, multiply coordinates by 4.
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(imgOutput, last_face_label, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

    # ================= SAFEHOUSE: unknown person hold detection (one-time alert + limited capture) =================
    # Declare these at the top of your script (near globals):
    # captured_once = False
    # captured_count = 0

    if safehouse_mode:
        if last_face_label == "Unknown":
            if unknown_start_time == 0:
                unknown_start_time = time.time()
            elif time.time() - unknown_start_time >= unknown_hold_duration:  # time.time() --> retruns current time in seconds
                if not captured_once:
                    set_status("UNKNOWN detected during SafeHouse Mode!", 6)
                    threading.Thread(target=safe_play, args=(siren_path,), daemon=True).start()

                    # ---- Create folder if not exists ----
                    capture_dir = os.path.join(os.getcwd(), "Captured_Frames")
                    os.makedirs(capture_dir, exist_ok=True)

                    # ---- Capture up to 5 images ----
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    for i in range(5):
                        filename = os.path.join(capture_dir, f"Unknown_{timestamp}_{i+1}.jpg")
                        cv2.imwrite(filename, imgOutput)
                        time.sleep(0.3)  # small delay between captures

                    # ---- Send alerts only once ----
                    async_send_sms(owner_number, "Unknown person detected during SafeHouse Mode! 5 images captured.")
                    async_send_sms(police_number, " Possible intrusion detected at SafeHouse.")
                    async_make_call(owner_number, "Unknown person detected during SafeHouse Mode. Authorities have been notified.")

                    captured_once = True  # Ensures alerts are sent only once until a known person resets it.
        else:
            # As soon as a known person is seen, reset everything.
            unknown_start_time = 0
            captured_once = False


    # ================= HAND PROCESSING & CLASSIFICATION =================
    if detected_region and region_type == "hand":
        x, y, w, h = detected_region
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255    # Creates a 300×300 white image.
                                                                     # This will hold the cropped and resized hand.
        # Extracts hand region from frame with a small margin (offset = 20).
        # Prevents cutting off edges.
        y1, y2 = max(0, y - offset), min(frame_h, y + h + offset)
        x1, x2 = max(0, x - offset), min(frame_w, x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            # ---- Exp 4–5 : Grayscale Conversion (Filtering / Sharpening) ----
            # Simplifies processing (no color needed for gestures).
            gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
             # ---- Exp 2 : Contrast Stretching ----
             # It makes dark areas darker and bright areas brighter, so that the image looks clearer and more detailed.
            min_val, max_val = np.min(gray), np.max(gray)   # np.min(gray) --> minimum pixel intensity value in the grayscale image. darkest pixel
                                                            # np.max(gray) --> Finds the maximum pixel intensity (the brightest pixel). 
            if max_val - min_val > 0:
                gray = ((gray - min_val) / (max_val - min_val)) * 255 
            gray = np.uint8(gray)                           # Pixel values are truncated to integers between 0 and 255.
            # ---- Exp 1 : Power Law Transformation (Gamma Correction) ----
            #image enhancement techniques in computer vision.
            # image brightness non-linearly.
            gamma = 1.5      #(gamma) = 1.5, it darkens the bright areas of the image
                             # gamma = 0.5 brightens dark areas of the image,
            gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')
             # ---- Exp 8 : Edge Detection (Canny) ----
            edges = cv2.Canny(gray, 100, 200) # Canny Edge Detection algorithm , it requires only grayscale image as it works in intensity value
                                              # Any pixel stronger than 200 is definitely an edge.
                                              # Any pixel weaker than 100 is definitely not an edge.
            # ---- Convert Back for Display / Classification ----
            imgEnhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            aspectRatio = h / w
            try:
                if aspectRatio > 1:  # vertical hand
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:                # horizontal hand
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
            except Exception:
                pass

            cv2.imshow('ImageCrop_Enhanced', imgEnhanced)
            cv2.imshow('ImageWhite', imgWhite)

            try:
                prediction, index = classifier.getPrediction(imgWhite, draw=False)  # Sends the processed image to the trained gesture classifier (from Teachable Machine or custom model).
                                                                                    # prediction: confidence scores for all classes
                                                                                    # index: index of the class with highest probability
                if index < len(labels):      
                    label = labels[index]       # Converts index → actual label (e.g., “Help”, “ThumbsUp”, etc.)
            except Exception:                   # Avoids crash if prediction fails.
                pass

    # ================= DRAWING =================
    if detected_region:
        x, y, w, h = detected_region
        color = (0, 255, 0) if region_type == "hand" else (255, 0, 0)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), color, 3)
        label_text = label if label is not None else ""
        if region_type == "face":
            cv2.putText(imgOutput, f"FACE: {label_text}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(imgOutput, f"HAND: {label_text}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if label:
        cv2.putText(imgOutput, f"Detected: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    if label and region_type == "hand":
        current_time = time.time()

        # Reset timer if a new gesture appears
        if label != last_label:
            last_label = label
            gesture_start_time = current_time
            triggered = False

        elapsed = current_time - gesture_start_time

        # Only if the same gesture is held continuously
        # progress on the top
        if elapsed < hold_duration:
            pct = elapsed / hold_duration
            cv2.rectangle(imgOutput, (10, 80), (int(10 + 200 * pct), 100), (0, 255, 0), -1)
            cv2.rectangle(imgOutput, (10, 80), (210, 100), (255, 255, 255), 2)
            remaining = max(0, int(hold_duration - elapsed))
            cv2.putText(imgOutput, f"Holding {remaining}s", (220, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif not triggered:
            triggered = True
            set_status(f"{label} gesture triggered after 3s", 4)
            trigger_actions(label)
    else:
        # Resets everything if hand not visible anymore.
        last_label = None
        gesture_start_time = 0
        triggered = False



    if safehouse_mode:
        cv2.putText(imgOutput, "SAFEHOUSE MODE ON", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if last_face_label == "Unknown" and unknown_start_time != 0:
            remaining = max(0, int(unknown_hold_duration - (time.time() - unknown_start_time)))
            cv2.putText(imgOutput, f"Unknown hold: {remaining}s", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

    if status_text and time.time() < status_expire:
        cv2.putText(imgOutput, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    elif status_text and time.time() >= status_expire:
        status_text = ""

    cv2.namedWindow("SafeHomeCam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("SafeHomeCam", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("SafeHomeCam", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# from playsound import playsound    
# from twilio.rest import Client   
# import face_recognition
# import os
# import threading

# # ========================== CAMERA & DETECTORS ==========================
# cap = cv2.VideoCapture(0)
# hand_detector = HandDetector(maxHands=1)

# # ========================== FACE RECOGNITION ==========================
# faces_path = r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Data\faces\Pratham"
# images = []
# classNames = []
# if not os.path.exists(faces_path):
#     os.makedirs(faces_path) 

# myList = os.listdir(faces_path)
# for cl in myList:
#     curImg = cv2.imread(os.path.join(faces_path, cl))
#     if curImg is None:
#         continue
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodings = face_recognition.face_encodings(img)
#         if encodings:
#             encodeList.append(encodings[0])
#     return encodeList

# encodeListKnown = findEncodings(images)

# # ========================== GESTURE CLASSIFIER ==========================
# classifier = Classifier(
#     r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\keras_model.h5",
#     r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\labels.txt"
# )

# with open(r"C:\Users\prath\OneDrive\Desktop\SafeHomeCam\Model\labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

# labels = [l.split(maxsplit=1)[-1] if len(l.split()) > 1 else l for l in labels]

# offset = 20
# imgSize = 300

# # ========================== TWILIO SETUP (DISABLED) ==========================
# account_sid = "AC7798998dc09096e2fbd6c57e20127ff4"
# auth_token = "665b5bf18dd144876fa29393a1add0be"
# client = Client(account_sid, auth_token)
# twilio_number = "+12294588371"
# owner_number = "+918623083659"
# police_number = "+918623083659"  # "+919892207022"

# # ========================== ALERT & SAFEHOUSE SETTINGS ==========================
# last_label = None          
# last_face_label = None      
# gesture_start_time = 0
# triggered = False
# hold_duration = 3           
# frame_count = 0
# process_every_n_frames = 3
# status_text = ""
# status_expire = 0
# cooldown_seconds = 5
# last_trigger_time = {"Help":0, "Call":0, "Danger":0, "ThumbsUp":0, "ThumbsDown":0}

# alarm_path = os.path.join(os.getcwd(), "alarm.mp3")
# danger_path = os.path.join(os.getcwd(), "Danger.wav")
# siren_path = os.path.join(os.getcwd(), "siren.mp3")

# safehouse_mode = False
# unknown_start_time = 0
# unknown_hold_duration = 3  

# # ========================== UTILITY FUNCTIONS ==========================
# def safe_play(path):
#     if os.path.isfile(path):
#         try:
#             from playsound import playsound
#             playsound(path)
#         except:
#             pass

# def set_status(text, duration=3):
#     global status_text, status_expire
#     status_text = text
#     status_expire = time.time() + duration

# # ========================== UTILITY FUNCTIONS ==========================
# def safe_play(path):
#     if os.path.isfile(path):
#         try: playsound(path)
#         except: pass

# def send_sms_sync(to, message):
#     try:
#         msg = client.messages.create(body=message, from_=twilio_number, to=to)
#         return ("ok", getattr(msg, "sid", None))
#     except Exception as e:
#         return ("error", str(e))

# def make_call_sync(to, message):
#     try:
#         call = client.calls.create(twiml=f'<Response><Say>{message}</Say></Response>', from_=twilio_number, to=to)
#         return ("ok", getattr(call, "sid", None))
#     except Exception as e:
#         return ("error", str(e))

# def async_send_sms(to, message):
#     def job():
#         res = send_sms_sync(to, message)
#         set_status(f"SMS -> {to}: {res[0]}", 4)
#     threading.Thread(target=job, daemon=True).start()

# def async_make_call(to, message):
#     def job():
#         res = make_call_sync(to, message)
#         set_status(f"Call -> {to}: {res[0]}", 4)
#     threading.Thread(target=job, daemon=True).start()

# def set_status(text, duration=3):
#     global status_text, status_expire
#     status_text = text
#     status_expire = time.time() + duration

# # =======================================TRIGGER ACTIONS ===================================
# def trigger_actions(label):
#     global safehouse_mode, unknown_start_time, last_face_label

#     gesture_name = label.replace(" ", "").strip()
#     gesture_name = gesture_name.replace("ThumbsUp", "ThumbsUp").replace("ThumbsDown", "ThumbsDown")

#     if gesture_name not in ["Help", "Call", "Danger", "ThumbsUp", "ThumbsDown"]:
#         return

#     now = time.time()
#     if now - last_trigger_time.get(gesture_name, 0) < cooldown_seconds:
#         set_status(f"{gesture_name} (cooldown)", 2)
#         return
#     last_trigger_time[gesture_name] = now

#     # ============================ SAFEMODE CONTROL (only Pratham) ============================
#     if gesture_name == "ThumbsUp":
#         if last_face_label == "Pratham":
#             safehouse_mode = True
#             unknown_start_time = 0
#             set_status("SafeHouse Mode ON (Authorized: Pratham)", 5)
#         else:
#             set_status("Access Denied: Only Pratham can turn ON SafeHouse Mode", 5)

#     elif gesture_name == "ThumbsDown":
#         if last_face_label == "Pratham":
#             safehouse_mode = False
#             set_status("SafeHouse Mode OFF (Authorized: Pratham)", 5)
#             unknown_start_time = 0
#         else:
#             set_status("Access Denied: Only Pratham can turn OFF SafeHouse Mode", 5)

#     elif gesture_name == "Help":
#         set_status("HELP triggered", 5)
#         threading.Thread(target=safe_play, args=(alarm_path,), daemon=True).start()
#         async_send_sms(owner_number, "🚨 HELP detected! Immediate assistance may be required at your location.")

#     elif gesture_name == "Call":
#         set_status("CALL triggered", 5)
#         async_make_call(owner_number, "Emergency call request received. Please check on the situation immediately.")
#         async_send_sms(owner_number, "📞 CALL detected. The system has initiated a call to alert you.")

#     elif gesture_name == "Danger":
#         set_status("DANGER triggered", 6)
#         threading.Thread(target=safe_play, args=(danger_path,), daemon=True).start()
#         async_send_sms(owner_number, "⚠️ DANGER ALERT! Something unusual has been detected at your location.")
#         async_send_sms(police_number, "🚨 ALERT: Possible threat detected at the registered address.")
#         # async_make_call(police_number, "Emergency! A potential danger has been detected. Please respond immediately.")
#         async_make_call(owner_number, "Danger alert triggered! Authorities have been notified.")

# # ========================== MAIN LOOP ==========================
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#     imgOutput = img.copy()
#     frame_h, frame_w = img.shape[:2]

#     hands, img = hand_detector.findHands(img, flipType=False) 
#     detected_region = None
#     region_type = None
#     label = None

#     # ----- PRIORITIZE HAND if visible (so gestures are detected even if face present)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']  
#         detected_region = (x, y, w, h)
#         region_type = "hand"

#     # ----- If no hand, do face detection/recognition
#     if not hands:
#         imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#         imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#         facesCurFrame = face_recognition.face_locations(imgS)
#         encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#         if len(facesCurFrame) > 0:
#             # only process the first face
#             encodeFace = encodesCurFrame[0]
#             faceLoc = facesCurFrame[0]
#             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

#             if len(faceDis) > 0:
#                 matchIndex = np.argmin(faceDis)
#                 if matches[matchIndex] and faceDis[matchIndex] < 0.45:
#                     face_label = "Pratham"
#                 else:
#                     face_label = "Unknown"
#             else:
#                 face_label = "Unknown"

#             # scale face location back to original size
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             detected_region = (x1, y1, x2 - x1, y2 - y1)
#             region_type = "face"
#             label = face_label
#             last_face_label = face_label
#         else:
#             detected_region = None
#             region_type = None
#             last_face_label = None

#     # ================= SAFEHOUSE: unknown person hold detection (only when ON) =================
#     if safehouse_mode:
#         if last_face_label == "Unknown":
#             if unknown_start_time == 0:
#                 unknown_start_time = time.time()
#             elif time.time() - unknown_start_time >= unknown_hold_duration:
#                 # trigger one-time unknown alert (Twilio disabled)
#                 set_status("UNKNOWN detected during SafeHouse Mode!", 6)
#                 threading.Thread(target=safe_play, args=(siren_path,), daemon=True).start()
#                 async_send_sms(owner_number, "⚠️ Unknown person detected during SafeHouse Mode.")
#                 async_send_sms(police_number, "🚨 Possible intrusion detected at registered address.")
#                 async_make_call(owner_number, "Unknown person detected in SafeHouse Mode.")
#                 #async_make_call(police_number, "Emergency! Intruder detected.")
#                 unknown_start_time = 0
#         else:
#             unknown_start_time = 0

#     # ================= HAND PROCESSING & CLASSIFICATION =================
#     if detected_region and region_type == "hand":
#         x, y, w, h = detected_region
#         # create white square and place resized hand inside (same preprocessing used in your model)
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         y1, y2 = max(0, y - offset), min(frame_h, y + h + offset)
#         x1, x2 = max(0, x - offset), min(frame_w, x + w + offset)
#         imgCrop = img[y1:y2, x1:x2]

#         if imgCrop.size != 0:
#             gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

#             min_val, max_val = np.min(gray), np.max(gray)
#             if max_val - min_val > 0:
#                 gray = ((gray - min_val) / (max_val - min_val)) * 255
#             gray = np.uint8(gray)

#             gamma = 1.5      
#             gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

#             edges = cv2.Canny(gray, 100, 200)

#             # ---- Convert Back for Display / Classification ----
#             imgEnhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)



#             aspectRatio = h / w
#             try:
#                 if aspectRatio > 1:
#                     k = imgSize / h
#                     wCal = math.ceil(k * w)
#                     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                     wGap = math.ceil((imgSize - wCal) / 2)
#                     imgWhite[:, wGap:wCal + wGap] = imgResize
#                 else:
#                     k = imgSize / w
#                     hCal = math.ceil(k * h)
#                     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                     hGap = math.ceil((imgSize - hCal) / 2)
#                     imgWhite[hGap:hCal + hGap, :] = imgResize
#             except Exception:
#                 pass

#             cv2.imshow('ImageCrop_Enhanced', imgEnhanced)
#             cv2.imshow('ImageWhite', imgWhite)

#             try:
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 if index < len(labels):
#                     label = labels[index]
#             except Exception:
#                 pass

#     # ================= DRAWING (bounding boxes, labels, timers) =================
#     if detected_region:
#         x, y, w, h = detected_region
#         color = (0, 255, 0) if region_type == "hand" else (255, 0, 0)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), color, 3)

#         label_text = label if label is not None else ""
#         if region_type == "face":
#             cv2.putText(imgOutput, f"FACE: {label_text}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#         else:
#             cv2.putText(imgOutput, f"HAND: {label_text}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     if label:
#         cv2.putText(imgOutput, f"Detected: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

#     if label and region_type == "hand":
#         if label != last_label:
#             last_label = label
#             gesture_start_time = time.time()
#             triggered = False
#         else:
#             elapsed = time.time() - gesture_start_time
#             # show hold progress bar
#             if elapsed < hold_duration:
#                 pct = elapsed / hold_duration
#                 cv2.rectangle(imgOutput, (10, 80), (int(10 + 200 * pct), 100), (0,255,0), -1)
#                 cv2.rectangle(imgOutput, (10, 80), (210, 100), (255,255,255), 2)
#                 cv2.putText(imgOutput, f"Holding {int(elapsed)}s", (220, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#             if elapsed >= hold_duration and not triggered:
#                 triggered = True
#                 trigger_actions(label)

#     if safehouse_mode:
#         cv2.putText(imgOutput, "SAFEHOUSE MODE ON", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#         if last_face_label == "Unknown" and unknown_start_time != 0:
#             remaining = max(0, int(unknown_hold_duration - (time.time() - unknown_start_time)))
#             cv2.putText(imgOutput, f"Unknown hold: {remaining}s", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

#     # Status text (short messages)
#     if status_text and time.time() < status_expire:
#         cv2.putText(imgOutput, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#     elif status_text and time.time() >= status_expire:
#         status_text = ""

#     cv2.imshow("SafeHomeCam", imgOutput)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
