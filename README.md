# object-detection-and-scene-understanting-system
import cv2
from ultralytics import YOLO
import time
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import pyttsx3

# ----------------- Text-to-Speech (audio) -----------------
engine = pyttsx3.init('sapi5')  # use Windows speech engine
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

def speak(text: str):
    """Speak text synchronously (simpler & more reliable)."""
    try:
        print("[TTS] Speaking:", text)
        engine.say(text)
        engine.runAndWait()
        print("[TTS] Done.")
    except Exception as e:
        print("[TTS ERROR]", e)

# ----------------- Device -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------- YOLO for object detection -----------------
yolo_model = YOLO("yolov8n.pt")  # small & fast; downloads weights on first run

# ----------------- Emotion model -----------------
emotion_model_name = "trpakov/vit-face-expression"
emotion_processor = AutoImageProcessor.from_pretrained(emotion_model_name)
emotion_model = AutoModelForImageClassification.from_pretrained(emotion_model_name).to(device)

# OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------- Webcam -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not detected.")
    exit()

print("Auto scene + emotion description running (WITH AUDIO).")
print("Press 'q' to quit.")
speak("Scene and emotion description system started. Please hold one person and some objects in front of the camera.")

last_sentence = "Hold one person and some objects in front of the camera."
last_spoken = ""
last_time = 0.0
INTERVAL = 7.0  # seconds between scene descriptions

# ----------------- Helper: get emotion for largest face -----------------
def get_emotion(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = frame_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    inputs = emotion_processor(images=face_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    labels = emotion_model.config.id2label

    idx = int(np.argmax(probs))
    raw_emotion = labels[idx].lower()

    if "happy" in raw_emotion:
        return "happy"
    if "sad" in raw_emotion:
        return "sad"
    if "angry" in raw_emotion:
        return "angry"
    if "surprise" in raw_emotion:
        return "surprised"
    if "fear" in raw_emotion:
        return "afraid"
    if "disgust" in raw_emotion:
        return "disgusted"
    return "neutral"

# ----------------- Helper: build nice scene description -----------------
def build_scene_description(objects, emotion, person_count):
    main_objects = [o for o in objects if o != "person"]
    parts = []

    if person_count == 0 and not main_objects:
        return "No clear objects in the scene."

    if person_count > 0:
        emo = None if emotion is None else emotion
        if person_count == 1:
            if emo and emo != "neutral":
                parts.append(f"A {emo} person")
            else:
                parts.append("A person")
        else:
            if emo and emo != "neutral":
                parts.append(f"{emo} people")
            else:
                parts.append("People")

    if main_objects:
        objs = list(main_objects)
        if len(objs) == 1:
            obj_text = objs[0]
            if person_count > 0:
                parts.append(f"with a {obj_text}")
            else:
                parts.append(f"A {obj_text} in the scene")
        elif len(objs) == 2:
            obj_text = f"{objs[0]} and {objs[1]}"
            if person_count > 0:
                parts.append(f"with {obj_text}")
            else:
                parts.append(f"{obj_text} in the scene")
        else:
            obj_text = ", ".join(objs[:-1]) + f" and {objs[-1]}"
            if person_count > 0:
                parts.append(f"with {obj_text}")
            else:
                parts.append(f"{obj_text} in the scene")

    description = " ".join(parts).strip()
    if not description:
        description = "A simple scene."

    if not description.endswith("."):
        description += "."

    return description

# ----------------- Main loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()

    det_results = yolo_model(frame, conf=0.25, verbose=False)
    names = yolo_model.names

    detected = []
    for box in det_results[0].boxes:
        cls_id = int(box.cls[0].item())
        label = names.get(cls_id, "object")
        detected.append(label)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    person_count = detected.count("person")
    unique_objects = sorted(set(detected))

    if now - last_time > INTERVAL:
        last_time = now

        emotion = None
        if person_count > 0:
            emotion = get_emotion(frame)

        sentence = build_scene_description(unique_objects, emotion, person_count)

        if sentence != last_spoken:
            last_spoken = sentence
            print("Scene:", sentence)
            # 🔊 speak the description
            speak(sentence)

        last_sentence = sentence

    cv2.putText(frame, last_sentence, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Scene + Emotion Description", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
