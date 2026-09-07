# -*- coding: utf-8 -*-
"""
Aruco_detection_V2_USB.py
==============================
MAIN FILE — run this.

Full integration:
  1. ArUco boundary detection  (ArucoDetection_definitions.py)
  2. YOLOv8 object detection   (ultralytics)
  3. Voice control             (Google STT)
  4. Robot arm pick-and-place  (arm_control.py + solverNNA.py)

FOLDER STRUCTURE:
    your_folder/
    ├── Aruco_detection_V2_USB.py       ← RUN THIS
    ├── ArucoDetection_definitions.py   ← boundary helper
    ├── arm_control.py                  ← arm control
    └── solverNNA.py                    ← inverse kinematics

INSTALL PACKAGES:
    pip install opencv-contrib-python numpy
    pip install ultralytics
    pip install sounddevice soundfile scipy SpeechRecognition pyserial

BEFORE RUNNING:
    1. Upload arduino_mg995_arm.ino to Arduino
    2. Set COM_PORT in arm_control.py
    3. Measure arm link lengths → update solverNNA.py
    4. Set WORKSPACE_WIDTH_MM and WORKSPACE_HEIGHT_MM below
    5. Place 4x ArUco markers (IDs 1,2,3,4) at workspace corners

CONTROLS:
    CLICK blue button  → speak object name (3 seconds)
    Say "scissors"     → arm picks up scissors
    Say "pick phone"   → arm picks up phone
    C key              → clear voice target
    + / - keys         → confidence up/down
    T key              → type new object list
    Q key              → quit and home arm
"""

import cv2
import numpy as np
import threading
import time
import sounddevice as sd
import soundfile
from scipy.io.wavfile import write as wav_write
import speech_recognition as sr

# ── Boundary helper ───────────────────────────────────────────────
from ArucoDetection_definitions import (
    getMarkerCoordinates,
    draw_corners,
    draw_numbers,
    show_spec,
    draw_field,
    four_point_transform
)

# ── Arm control ───────────────────────────────────────────────────
import arm_control

# ══════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════

CAMERA_INDEX  = 1        # 1=USB cam, 0=laptop cam
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
DICT_BOUNDARY = "DICT_4X4_50"

# ── Workspace size in mm — measure your actual workspace ──────────
WORKSPACE_WIDTH_MM  = 300   # ← tune
WORKSPACE_HEIGHT_MM = 300   # ← tune

# ── Objects to detect ─────────────────────────────────────────────
DETECT_OBJECTS = [
    "pen", "pencil",
    "scissors",
    "phone", "mobile phone",
    "cup", "bottle",
    "book",
    "keyboard", "mouse", "laptop",
    "remote", "keys",
    "person",
]

CONFIDENCE   = 0.25
DETECT_EVERY = 2      # run YOLO every N frames

# ── Voice button on Workspace window ─────────────────────────────
BTN_X1, BTN_Y1 = 10,  10
BTN_X2, BTN_Y2 = 420, 60

# ── Audio ─────────────────────────────────────────────────────────
AUDIO_RATE = 44100
AUDIO_SEC  = 3
AUDIO_TMP1 = "tmp_voice.wav"
AUDIO_TMP2 = "tmp_voice_pcm.wav"

# ── Pick cooldown — seconds between picks ────────────────────────
PICK_COOLDOWN = 6

ARUCO_DICT_MAP = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":  cv2.aruco.DICT_6X6_50,
}

PALETTE = [
    (0,255,0),(0,200,255),(255,100,0),(180,0,255),
    (0,255,180),(255,200,0),(0,100,255),(255,0,100),
]

# ── State ─────────────────────────────────────────────────────────
voice_target   = ""
voice_active   = False
recording_now  = False
arm_busy       = False
last_pick_time = 0


# ══════════════════════════════════════════════════════════════════
#  YOLO DETECTOR
# ══════════════════════════════════════════════════════════════════

class YOLOWorldDetector:
    def __init__(self):
        from ultralytics import YOLOWorld
        print("[YOLO] Loading YOLO-World...")
        self._model = YOLOWorld("yolov8s-worldv2.pt")
        print("[YOLO] YOLO-World ready.")

    def detect(self, frame, labels, conf):
        self._model.set_classes(labels)
        results = self._model.predict(frame, conf=conf, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cid  = int(box.cls[0])
            name = labels[cid] if cid < len(labels) else "object"
            dets.append({
                "name": name,
                "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                "bbox": (x1,y1,x2-x1,y2-y1),
                "conf": float(box.conf[0])
            })
        return dets


class YOLOv8Detector:
    def __init__(self):
        from ultralytics import YOLO
        print("[YOLO] Loading YOLOv8n...")
        self._model = YOLO("yolov8n.pt")
        print("[YOLO] YOLOv8n ready.")

    def detect(self, frame, labels, conf):
        results = self._model.predict(frame, conf=conf, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            name = self._model.names[int(box.cls[0])]
            dets.append({
                "name": name,
                "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                "bbox": (x1,y1,x2-x1,y2-y1),
                "conf": float(box.conf[0])
            })
        return dets


def load_detector():
    for label, cls in [("YOLO-World", YOLOWorldDetector),
                       ("YOLOv8n",    YOLOv8Detector)]:
        try:
            print(f"[YOLO] Trying {label}...")
            det = cls()
            return det, label
        except Exception as e:
            print(f"[YOLO] {label} failed: {e}")
    return None, "none"


# ══════════════════════════════════════════════════════════════════
#  PIXEL → MM CONVERSION
# ══════════════════════════════════════════════════════════════════

def pixel_to_mm(cx, cy, ws_w, ws_h):
    """
    Convert pixel coordinate inside workspace to mm coordinate
    relative to arm base origin.

    Assumes workspace origin (0,0 pixel) maps to arm coordinate
    system centre. Adjust offsets to match your physical setup.
    """
    x_mm = int((cx / ws_w) * WORKSPACE_WIDTH_MM)
    y_mm = int((cy / ws_h) * WORKSPACE_HEIGHT_MM)

    # Apply camera perspective compensation
    x_comp, y_comp = arm_control.camera_compensation(x_mm, y_mm)
    return x_comp, y_comp


# ══════════════════════════════════════════════════════════════════
#  ARM PICK THREAD
# ══════════════════════════════════════════════════════════════════

def arm_pick_thread(x_mm, y_mm, obj_name):
    """Run arm pick-and-place in background so camera keeps running."""
    global arm_busy, voice_target, voice_active, last_pick_time
    arm_busy = True
    print(f"\n[ARM] Starting pick: '{obj_name}' at ({x_mm},{y_mm}) mm")
    try:
        arm_control.pick_up(x_mm, y_mm)
    except Exception as e:
        print(f"[ARM] Error during pick: {e}")
    finally:
        arm_busy       = False
        voice_target   = ""
        voice_active   = False
        last_pick_time = time.time()
        print("[ARM] Pick complete. Ready for next command.")


def start_arm_pick(x_mm, y_mm, obj_name):
    t = threading.Thread(
        target=arm_pick_thread,
        args=(x_mm, y_mm, obj_name),
        daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════
#  VOICE RECOGNITION
# ══════════════════════════════════════════════════════════════════

def record_and_transcribe():
    global voice_target, voice_active, recording_now
    recording_now = True
    try:
        print("[VOICE] Recording 3 seconds... speak now!")
        audio = sd.rec(int(AUDIO_SEC * AUDIO_RATE),
                       samplerate=AUDIO_RATE, channels=1)
        sd.wait()
        wav_write(AUDIO_TMP1, AUDIO_RATE, audio)

        data, rate = soundfile.read(AUDIO_TMP1)
        soundfile.write(AUDIO_TMP2, data, rate, subtype='PCM_16')

        recog = sr.Recognizer()
        with sr.AudioFile(AUDIO_TMP2) as source:
            audio_data = recog.record(source)
        text = recog.recognize_google(audio_data).lower()
        print(f"[VOICE] Heard: '{text}'")

        fillers = {"pick","up","the","a","an","get","grab",
                   "take","fetch","please","me","that","find",
                   "show","highlight","detect","i","want"}
        words  = [w for w in text.split() if w not in fillers]
        target = " ".join(words).strip()

        if target:
            voice_target = target
            voice_active = True
            print(f"[VOICE] Looking for: '{voice_target}'")
        else:
            print("[VOICE] Could not extract object name.")

    except sr.UnknownValueError:
        print("[VOICE] Could not understand. Try again.")
    except sr.RequestError as e:
        print(f"[VOICE] STT error: {e}")
    except Exception as e:
        print(f"[VOICE] Error: {e}")
    finally:
        recording_now = False


def start_voice_thread():
    t = threading.Thread(target=record_and_transcribe, daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════
#  MOUSE CALLBACK
# ══════════════════════════════════════════════════════════════════

def on_mouse(event, x, y, flags, params):
    global voice_target, voice_active
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            if not recording_now and not arm_busy:
                start_voice_thread()
            elif arm_busy:
                print("[VOICE] Arm is busy, wait...")
        else:
            voice_target = ""
            voice_active = False


# ══════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════

def get_color(name):
    return PALETTE[hash(name) % len(PALETTE)]


def draw_voice_button(frame):
    if arm_busy:
        color = (0, 100, 100)
        label = "  ARM MOVING... please wait"
    elif recording_now:
        color = (0, 180, 0)
        label = "  RECORDING... SPEAK NOW!"
    else:
        color = (0, 60, 200)
        label = "  CLICK HERE  then say object name"

    cv2.rectangle(frame, (BTN_X1,BTN_Y1), (BTN_X2,BTN_Y2), color, -1)
    cv2.rectangle(frame, (BTN_X1,BTN_Y1), (BTN_X2,BTN_Y2), (255,255,255), 1)
    cv2.putText(frame, label,
                (BTN_X1+8, BTN_Y1+36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def draw_voice_status(frame):
    if arm_busy:
        cv2.putText(frame, "ARM IS PICKING OBJECT...",
                    (BTN_X1, BTN_Y2+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 2)
    elif voice_active and voice_target:
        cv2.putText(frame, f"Target: '{voice_target}'  — arm will pick when found",
                    (BTN_X1, BTN_Y2+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    else:
        cv2.putText(frame, "Click button → say object name → arm picks it",
                    (BTN_X1, BTN_Y2+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)


def draw_all_detections(frame, detections, ws_w, ws_h):
    """
    Draw all detections.
    Matched object → GREEN box + triggers arm pick.
    Returns best matched detection or None.
    """
    best      = None
    best_conf = 0

    for det in detections:
        x, y, w, h = det["bbox"]
        name  = det["name"]
        conf  = det["conf"]
        color = get_color(name)

        matched = (
            voice_active and voice_target and not arm_busy and (
                voice_target in name or
                name in voice_target or
                any(t in name for t in voice_target.split())
            )
        )

        if matched:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 4)
            lbl = f"{name}  <<  TARGET  {conf:.0%}"
            cv2.rectangle(frame, (x,y-30), (x+len(lbl)*11,y), (0,180,0), -1)
            cv2.putText(frame, lbl, (x+2,y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
            cv2.circle(frame, (det["cx"],det["cy"]), 8, (0,255,0), -1)
            if conf > best_conf:
                best_conf = conf
                best = det
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            lbl = f"{name} {conf:.0%}"
            (lw,lh),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x,y-lh-8), (x+lw+4,y), color, -1)
            cv2.putText(frame, lbl, (x+2,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.circle(frame, (det["cx"],det["cy"]), 5, (0,0,255), -1)

    return best


def draw_status_bar(frame, detections, sq_found, det_name, conf):
    fh = frame.shape[0]
    arm_status = "BUSY" if arm_busy else "READY"
    lines = [
        f"Q=quit  C=clear  +/-=conf  T=objects",
        f"Arm: {arm_status}   Detector: {det_name}   Conf: {conf:.2f}",
        f"Boundary: {'OK' if sq_found else 'waiting for markers 1-4'}  |  {len(detections)} detected",
    ]
    for i, line in enumerate(reversed(lines)):
        cv2.putText(frame, line,
                    (10, fh-10-i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,255), 1)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    global CONFIDENCE, DETECT_OBJECTS, voice_target, voice_active

    # ── Load YOLO ─────────────────────────────────────────────────
    detector, det_name = load_detector()
    det_loaded = detector is not None

    # ── Load ArUco ────────────────────────────────────────────────
    print("[INFO] Initializing ArUco...")
    aruco_dict   = cv2.aruco.getPredefinedDictionary(
                       ARUCO_DICT_MAP[DICT_BOUNDARY])
    aruco_params = cv2.aruco.DetectorParameters()

    # ── Open camera ───────────────────────────────────────────────
    print(f"[INFO] Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_INDEX} not found. Try 0 or 2.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {aw}x{ah}")

    # ── Window + mouse ────────────────────────────────────────────
    WIN = "Workspace"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    print()
    print("=" * 52)
    print("  FULL SYSTEM READY")
    print("=" * 52)
    print("  1. Place markers 1,2,3,4 at workspace corners")
    print("  2. Put objects inside workspace")
    print("  3. CLICK BLUE BUTTON → say object name")
    print("  4. ARM AUTOMATICALLY PICKS THE OBJECT")
    print("  Q=quit  C=clear voice  +/-=confidence")
    print("=" * 52)
    print()

    # Fallback boundary
    current_sq = [
        [10,    ah-10],
        [aw-10, ah-10],
        [aw-10, 10   ],
        [10,    10   ]
    ]
    sq_pts      = current_sq[:]
    sq_found    = False
    frame_count = 0
    last_dets   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame_clean = frame.copy()
        frame_count += 1

        # ── ArUco boundary ────────────────────────────────────────
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        markers, ids, _ = aruco_detector.detectMarkers(frame)
        ids_list = [int(i[0]) for i in ids] if ids is not None else None

        corners, cids = getMarkerCoordinates(markers, ids_list, 0)

        if cids is not None:
            cnt = 0
            for mid in cids:
                if 1 <= mid <= 4:
                    current_sq[mid-1] = corners[cnt]
                cnt += 1
        corners = current_sq
        cids    = [1, 2, 3, 4]

        cv2.aruco.drawDetectedMarkers(frame, markers)
        draw_corners(frame, corners)
        draw_numbers(frame, corners, cids)
        show_spec(frame, corners)
        frame_boundary, sq_found = draw_field(frame, corners, cids)

        if sq_found:
            sq_pts = corners

        # ── Perspective crop ──────────────────────────────────────
        workspace = four_point_transform(frame_clean, np.array(sq_pts))
        if workspace is None or workspace.size == 0:
            cv2.imshow("Boundary View", frame_boundary)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        ws_h, ws_w = workspace.shape[:2]

        # ── YOLO detection ────────────────────────────────────────
        if det_loaded and frame_count % DETECT_EVERY == 0:
            try:
                last_dets = detector.detect(
                    workspace, DETECT_OBJECTS, CONFIDENCE)
            except Exception as e:
                print(f"[DETECT] Error: {e}")
                last_dets = []

        # ── Draw detections ───────────────────────────────────────
        best = draw_all_detections(workspace, last_dets, ws_w, ws_h)
        draw_voice_button(workspace)
        draw_voice_status(workspace)
        draw_status_bar(workspace, last_dets, sq_found, det_name, CONFIDENCE)

        # ── TRIGGER ARM if match found ────────────────────────────
        cooldown_ok = (time.time() - last_pick_time) > PICK_COOLDOWN
        if best and not arm_busy and cooldown_ok:
            cx, cy   = best["cx"], best["cy"]
            x_mm, y_mm = pixel_to_mm(cx, cy, ws_w, ws_h)
            print(f"\n[MATCH] '{best['name']}'  pixel:({cx},{cy})  mm:({x_mm},{y_mm})")
            start_arm_pick(x_mm, y_mm, best["name"])

        # Console status
        if last_dets:
            names = [f"{d['name']}({d['conf']:.0%})" for d in last_dets]
            print(f"  Detected: {', '.join(names)}       ", end="\r")

        # ── Show windows ──────────────────────────────────────────
        cv2.imshow("Boundary View", frame_boundary)
        cv2.imshow(WIN,             workspace)

        # ── Keys ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quitting — homing arm...")
            if not arm_busy:
                arm_control.home()
            break

        elif key == ord('c'):
            voice_target = ""
            voice_active = False
            print("\n[VOICE] Cleared.")

        elif key in (ord('+'), ord('=')):
            CONFIDENCE = min(0.95, round(CONFIDENCE+0.05, 2))
            print(f"\n[INFO] Confidence → {CONFIDENCE:.2f}")

        elif key == ord('-'):
            CONFIDENCE = max(0.05, round(CONFIDENCE-0.05, 2))
            print(f"\n[INFO] Confidence → {CONFIDENCE:.2f}")

        elif key == ord('t'):
            print("\n[INPUT] Objects (comma separated): ",
                  end="", flush=True)
            user_input = input()
            if user_input.strip():
                DETECT_OBJECTS = [
                    o.strip() for o in user_input.split(",")
                    if o.strip()]
                print(f"[INFO] Now detecting: {DETECT_OBJECTS}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == '__main__':
    main()
