"""
SafeHomeCam — Flask Web Dashboard
Wraps Main_app.py's process_frame() to stream live video,
serve real-time status, event logs, and captured frames.
"""

import cv2
import csv
import os
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
import pyttsx3
import uuid

# ---------------------------------------------------------------------------
# Import the existing backend (Main_app.py) — all detection logic lives there
# ---------------------------------------------------------------------------
import Main_app

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(BASE_DIR, "Captured_Frames")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
LOG_FILE = os.path.join(BASE_DIR, "gesture_log.csv")
REMINDERS_FILE = os.path.join(BASE_DIR, "reminders.json")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
reminders = []
def load_reminders():
    global reminders
    if os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE, 'r') as f:
                reminders = json.load(f)
        except Exception as e:
            print(f"Error loading reminders: {e}")
            reminders = []

def save_reminders():
    try:
        with open(REMINDERS_FILE, 'w') as f:
            json.dump(reminders, f, indent=4)
    except Exception as e:
        print(f"Error saving reminders: {e}")

load_reminders()

# Camera Toggle state
camera_enabled = True

# TTS Engine initialization
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()

def speak(text):
    def _speak():
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

latest_status = {
    "gesture": None,
    "face": None,
    "safehouse": False,
    "safe_voice_mode": False,
    "camera_enabled": True,
    "status": "",
    "hold_pct": 0.0,
    "fps": 0,
}
lock = threading.Lock()
new_frame_event = threading.Event()

# Frame buffers — separated so camera read never blocks on AI
latest_raw_frame = None        # always the newest camera frame
latest_ai_result = {"gesture": None, "face": None, "safehouse": False, "status": ""}
output_frame = None
frame_counter = 0              # bumped on each new composed frame
frame_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Thread 1: Camera Reader — grabs frames as fast as possible
# ---------------------------------------------------------------------------
def camera_reader_thread():
    """Read camera frames continuously. Never does heavy processing."""
    global latest_raw_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[app] ERROR: Could not open camera.")
        return

    print("[app] Camera opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_raw_frame = frame

# ---------------------------------------------------------------------------
# Thread 2: AI Processor — runs detection at its own pace
# ---------------------------------------------------------------------------
def ai_processor_thread():
    """Runs process_frame() on the latest camera frame, without blocking video."""
    global latest_ai_result

    while True:
        with frame_lock:
            frame = latest_raw_frame

        if frame is None:
            time.sleep(0.01)
            continue

        # This is the heavy call (gesture + face recognition)
        result = Main_app.process_frame(frame.copy())

        with lock:
            latest_ai_result = result

        # Don't spin faster than needed; ~10-15 detections/sec is plenty
        time.sleep(0.03)

# ---------------------------------------------------------------------------
# Thread 4: Reminder Scheduler — checks for medicine reminders
# ---------------------------------------------------------------------------
def reminder_scheduler_thread():
    """Checks for active reminders every minute."""
    global reminders
    last_check_minute = -1

    while True:
        now = datetime.now()
        current_minute = now.minute

        if current_minute != last_check_minute:
            last_check_minute = current_minute
            current_time_str = now.strftime("%H:%M")
            triggered_indices = []

            with lock:
                for i, r in enumerate(reminders):
                    if r['time'] == current_time_str:
                        # Logic for recurrence
                        should_trigger = False
                        
                        created_at = datetime.fromisoformat(r['created_at'])
                        days_elapsed = (now - created_at).days

                        if r['recurrence'] == 'once':
                            should_trigger = True
                            triggered_indices.append(i)
                        elif r['recurrence'] == 'daily':
                            should_trigger = True
                        elif r['recurrence'] == 'weekly':
                            if days_elapsed < 7:
                                should_trigger = True
                            else:
                                triggered_indices.append(i)

                        if should_trigger:
                            # 1. Voice Announcement
                            msg = f"Reminder: Time for your medicine, {r['name']}."
                            speak(msg)
                            
                            # 2. Twilio SMS
                            sms_msg = f"💊 Medicine Reminder: It's time to take {r['name']}."
                            Main_app.async_send_sms(Main_app.caretaker_number, sms_msg)
                            
                            # 3. Log it
                            Main_app.log_event("Reminder", "System", f"Triggered reminder: {r['name']}")

                # Remove 'once' or expired 'weekly' reminders
                if triggered_indices:
                    reminders = [r for idx, r in enumerate(reminders) if idx not in triggered_indices]
                    save_reminders()

        time.sleep(10) # Check every 10 seconds to catch the minute change accurately

# ---------------------------------------------------------------------------
# Thread 3: Frame Composer — renders overlays and encodes JPEG
# ---------------------------------------------------------------------------
def frame_composer_thread():
    """Combines latest raw frame + latest AI result into the MJPEG output."""
    global output_frame, latest_status, frame_counter

    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0

    while True:
        with frame_lock:
            frame = latest_raw_frame

        if frame is None:
            time.sleep(0.005)
            continue

        with lock:
            result = dict(latest_ai_result)

        display = frame.copy()
        h, w = display.shape[:2]

        # Face bounding box
        if Main_app.last_face_location is not None:
            y1, x2, y2, x1 = Main_app.last_face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            face_name = Main_app.last_face_label or "Unknown"
            box_color = (0, 0, 255) if face_name == "Unknown" else (0, 255, 136)

            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            label_size = cv2.getTextSize(face_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display, (x1, y2), (x1 + label_size[0] + 10, y2 + label_size[1] + 14), box_color, -1)
            cv2.putText(display, face_name, (x1 + 5, y2 + label_size[1] + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Gesture label
        if result["gesture"]:
            cv2.putText(display, f"Gesture: {result['gesture']}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2, cv2.LINE_AA)

        # Hold progress bar
        hold_pct = 0.0
        if Main_app.last_label and not Main_app.triggered:
            elapsed = time.time() - Main_app.gesture_start_time
            if elapsed < Main_app.hold_duration:
                hold_pct = elapsed / Main_app.hold_duration
                bar_w = int(300 * hold_pct)
                cv2.rectangle(display, (15, 50), (15 + bar_w, 65), (0, 255, 136), -1)
                cv2.rectangle(display, (15, 50), (315, 65), (100, 100, 100), 2)
                remaining = max(0, int(Main_app.hold_duration - elapsed))
                cv2.putText(display, f"Hold: {remaining}s", (325, 63),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # SafeHouse badge
        if result["safehouse"]:
            cv2.putText(display, "SAFEHOUSE MODE", (w - 250, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2, cv2.LINE_AA)

        # Camera Disabled Placeholder
        if not camera_enabled:
            # Create a black frame with text
            display = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(display, "CAMERA DISABLED", (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(display, "Click 'Camera ON' to resume", (w // 2 - 120, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            # Override AI result display when disabled
            result["gesture"] = None
            result["face"] = None

        # SafeVoiceMode badge
        if Main_app.safe_voice_mode:
            cv2.putText(display, "VOICE MODE", (w - 220, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        # Status text
        if result["status"]:
            cv2.putText(display, result["status"], (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

        # FPS
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(display, f"FPS: {current_fps}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Update shared status for API
        with lock:
            latest_status["gesture"] = result["gesture"]
            latest_status["face"] = result["face"]
            latest_status["safehouse"] = result["safehouse"]
            latest_status["safe_voice_mode"] = Main_app.safe_voice_mode
            latest_status["camera_enabled"] = camera_enabled
            latest_status["status"] = result["status"]
            latest_status["hold_pct"] = hold_pct
            latest_status["fps"] = current_fps

        # Encode JPEG
        success, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            with lock:
                output_frame = buffer.tobytes()
            frame_counter += 1
            new_frame_event.set()
        
        time.sleep(0.01)


def generate_mjpeg():
    """Yield MJPEG frames for the /video_feed endpoint."""
    last_sent = -1
    print("[app] MJPEG stream: Client connected.")
    try:
        while True:
            # Wait for a new frame from the composer
            if not new_frame_event.wait(timeout=1.0):
                continue # No new frame, keep waiting
            
            # We don't clear() here because multiple clients might be watching.
            # Instead, the composer sets it, and we check the counter.

            current = frame_counter
            if output_frame is None or current == last_sent:
                time.sleep(0.01)
                continue

            last_sent = current
            with lock:
                frame_bytes = output_frame
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        print("[app] MJPEG stream: Client disconnected.")
    except Exception as e:
        print(f"[app] MJPEG stream: Error {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    with lock:
        return jsonify(latest_status)


@app.route('/api/logs')
def api_logs():
    """Return the last 50 log entries as JSON."""
    entries = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in rows[-50:]:
                    entries.append(row)
        except Exception:
            pass
    entries.reverse()
    return jsonify(entries)


@app.route('/api/captures')
def api_captures():
    """Return a list of captured frame filenames."""
    files = []
    if os.path.exists(CAPTURE_DIR):
        for fname in sorted(os.listdir(CAPTURE_DIR), reverse=True):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(fname)
    return jsonify(files[:30])


@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURE_DIR, filename)


# ---------------------------------------------------------------------------
# SafeVoiceMode API
# ---------------------------------------------------------------------------
@app.route('/api/voice_command', methods=['POST'])
def api_voice_command():
    """Receive a voice command from the browser and delegate to Main_app."""
    data = request.get_json(force=True)
    command = data.get("command", "")
    if not command:
        return jsonify({"error": "No command provided"}), 400

    result = Main_app.handle_voice_command(command)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Reminder API
# ---------------------------------------------------------------------------
@app.route('/api/reminders', methods=['GET', 'POST'])
def api_reminders():
    global reminders
    if request.method == 'GET':
        return jsonify(reminders)
    
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'name' not in data or 'time' not in data or 'recurrence' not in data:
            return jsonify({"error": "Missing fields"}), 400
        
        new_reminder = {
            "id": str(uuid.uuid4()),
            "name": data['name'],
            "time": data['time'],
            "recurrence": data['recurrence'],
            "created_at": datetime.now().isoformat()
        }
        
        with lock:
            reminders.append(new_reminder)
            save_reminders()
            
        return jsonify(new_reminder), 201

@app.route('/api/reminders/<reminder_id>', methods=['DELETE'])
def delete_reminder(reminder_id):
    global reminders
    with lock:
        reminders = [r for r in reminders if r['id'] != reminder_id]
        save_reminders()
    return jsonify({"status": "success"})


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------
@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({"activation_word": "blue mango"})
    
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'activation_word' not in data:
            return jsonify({"error": "Missing activation_word"}), 400
        
        # Save to settings.json
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Tell Main_app to reload
        Main_app.reload_settings()
        
        return jsonify({"status": "success", "activation_word": data['activation_word']})


# ---------------------------------------------------------------------------
# Camera Control API
# ---------------------------------------------------------------------------
@app.route('/api/camera/toggle', methods=['POST'])
def api_camera_toggle():
    global camera_enabled
    camera_enabled = not camera_enabled
    return jsonify({"camera_enabled": camera_enabled})


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Three separate threads — camera read, AI processing, frame composing
    threading.Thread(target=camera_reader_thread, daemon=True).start()
    threading.Thread(target=ai_processor_thread, daemon=True).start()
    threading.Thread(target=frame_composer_thread, daemon=True).start()
    threading.Thread(target=reminder_scheduler_thread, daemon=True).start()

    print("\n" + "=" * 50)
    print("  SafeHomeCam Dashboard")
    print("  Open http://localhost:5001 in your browser")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
