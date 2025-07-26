from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cheating_detector import detect_cheating_clips
import sounddevice as sd
import soundfile as sf
import cv2
import numpy as np
from keras.models import load_model
import threading
import time
import os
import random
import warnings
import uvicorn

warnings.filterwarnings("ignore")

# --- FastAPI App Setup ---
app = FastAPI()
os.makedirs("static", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("audio_recordings", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")
templates = Jinja2Templates(directory="templates")

# --- Load Model ---
try:
    model = load_model("newModel.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Globals ---
capture = None
camera_thread = None
recording_writer = None
recording_active = False
audio_active = False
frame_lock = threading.Lock()
current_frame = None

session = {}
QUESTION_BANK = {
    "web_developer": [
        "What is React?",
        "Difference between let, var, const?",
        "What is REST API?",
        "Explain promises in JS.",
        "How do you optimize a website?"
    ],
    "data_scientist": [
        "What is overfitting?",
        "Explain KNN.",
        "What is p-value?",
        "Explain feature scaling.",
        "Describe a past ML project."
    ]
}

# --- Audio Recording ---
def record_audio(audio_path):
    global audio_active
    samplerate = 44100
    channels = 2
    try:
        print(f"🎙️ Audio recording started: {audio_path}")
        audio_active = True
        with sf.SoundFile(audio_path, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as file:
            def callback(indata, frames, time, status):
                if status:
                    print(f"⚠️ Audio input status: {status}")
                if audio_active:
                    file.write(indata.copy())

            with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
                while audio_active:
                    sd.sleep(100)
    except Exception as e:
        print(f"❌ Audio recording error: {e}")

def start_recording(filename):
    global recording_writer, recording_active, audio_active
    try:
        video_path = f"recordings/{filename}.mp4"
        audio_path = f"audio_recordings/{filename}.wav"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ✅ Instead of 'avc1'
        recording_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))

        if recording_writer.isOpened():
            recording_active = True
            print(f"✅ Video recording started: {video_path}")

        audio_thread = threading.Thread(target=record_audio, args=(audio_path,), daemon=True)
        audio_thread.start()
    except Exception as e:
        print(f"❌ Error starting recording: {e}")
def stop_recording():
    global recording_writer, recording_active, audio_active
    if recording_writer:
        recording_writer.release()
        print("🛑 Video recording stopped.")
    recording_writer = None
    recording_active = False
    audio_active = False
    print("🛑 Audio recording stopped.")

# --- Camera Handling ---
def capture_frames():
    global capture, current_frame, recording_writer, recording_active
    while capture and capture.isOpened():
        success, frame = capture.read()
        if not success:
            continue
        frame = cv2.resize(frame, (640, 480))
        with frame_lock:
            current_frame = frame.copy()
        if recording_active and recording_writer:
            recording_writer.write(frame)
        time.sleep(0.01)

def start_camera():
    global capture, camera_thread
    if capture is None or not capture.isOpened():
        capture = cv2.VideoCapture(0)
        print("✅ Camera started")
        camera_thread = threading.Thread(target=capture_frames, daemon=True)
        camera_thread.start()

def stop_camera():
    global capture
    if capture and capture.isOpened():
        capture.release()
        print("🛑 Camera stopped")
    capture = None

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start")
def start(request: Request, role: str = Form(...)):
    questions = QUESTION_BANK.get(role, [])
    random.shuffle(questions)
    session["questions"] = questions
    session["current"] = 0
    session["role"] = role
    start_camera()
    return RedirectResponse(url="/interview", status_code=302)

@app.get("/interview")
def interview(request: Request):
    current = session.get("current", 0)
    questions = session.get("questions", [])
    stop_recording()

    if current >= len(questions):
        session["current"] = 0  # ✅ Reset to avoid infinite loop
        stop_camera()
        return RedirectResponse(url="/confidence-score", status_code=302)

    question = questions[current]
    stop_recording()
    start_recording(f"{session['role']}_q{current + 1}")

    return templates.TemplateResponse("interview.html", {
        "request": request,
        "question": question,
        "current": current + 1,
        "total": len(questions)
    })

@app.post("/next")
def next_question(request: Request):
    session["current"] += 1
    return RedirectResponse(url="/interview", status_code=302)

@app.get("/video")
def normal_cam():
    def generate():
        while True:
            with frame_lock:
                frame = current_frame.copy() if current_frame is not None else None
            if frame is None:
                continue
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/confidence-score", response_class=HTMLResponse)
def confidence_score(request: Request):
    role = session.get("role", "")
    questions = session.get("questions", [])
    total = len(questions)
    scores = []
    video_list = []

    for i in range(1, total + 1):
        video_filename = f"{role}_q{i}.mp4"
        video_path = f"recordings/{video_filename}"

        if os.path.exists(video_path):
            video_list.append(f"/recordings/{video_filename}")
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
            cap.release()

            if len(frames) > 0:
                frames_array = np.array(frames)
                preds = model.predict(frames_array, verbose=0)
                avg_confidence = np.mean(preds[:, 0]) * 100
                scores.append(round(avg_confidence, 2))
            else:
                scores.append(0)
        else:
            scores.append(0)

    overall_confidence = round(sum(scores) / len(scores), 2) if scores else 0
    if overall_confidence >= 80:
        test_result = "✅ Great Test! You were very confident."
    elif overall_confidence >= 50:
        test_result = "⚠️ Good Attempt! Try to be more confident."
    else:
        test_result = "❌ Low Confidence. Needs Improvement."

    try:
        from cheating_detector import detect_cheating_clips
        cheating_video_list = detect_cheating_clips(video_list)
    except Exception as e:
        print(f"❌ Cheating detection failed: {e}")

    return templates.TemplateResponse("confidence-score.html", {
        "request": request,
        "test_result": test_result,
        "overall_confidence": overall_confidence,
        "question_scores": list(zip(questions, scores)),
        "clips": video_list,
        "cheating_clips": cheating_video_list
    })

@app.get("/preview", response_class=HTMLResponse)
def preview(request: Request):
    stop_recording()
    stop_camera()
    role = session.get("role", "")
    questions = session.get("questions", [])
    total = len(questions)
    video_list = []
    audio_list = []

    for i in range(1, total + 1):
        video_filename = f"{role}_q{i}.mp4"
        audio_filename = f"{role}_q{i}.wav"
        if os.path.exists(f"recordings/{video_filename}"):
            video_list.append(f"/recordings/{video_filename}")
        if os.path.exists(f"audio_recordings/{audio_filename}"):
            audio_list.append(f"/audio_recordings/{audio_filename}")

    return templates.TemplateResponse("preview.html", {
        "request": request,
        "videos": video_list,
        "audios": audio_list
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
