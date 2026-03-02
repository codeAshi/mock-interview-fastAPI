# 🎥 Mock Interview System with Confidence Analysis (FastAPI)

A **mock interview platform** built with **FastAPI**, designed to simulate real interview scenarios.  
The system records **video & audio responses**, analyzes **confidence levels** using a deep learning model, and detects **possible cheating**.
  
---

## 🚀 Features

✅ **Role-based interview questions** (Data Scientist, Web Developer, etc.)  
✅ **Real-time video and audio recording** (using OpenCV & SoundDevice)  
✅ **Confidence analysis** using a pre-trained deep learning model (Keras/TensorFlow)  
✅ **Cheating detection** (face & eye movement analysis)  
✅ **Interactive confidence summary with charts & progress bars**  
✅ **Preview & download recorded clips after the test**  
✅ **Modern responsive UI built with HTML, CSS & Chart.js**

---

## 🛠️ Tech Stack

**Backend**: FastAPI, Uvicorn  
**Frontend**: HTML, CSS, JavaScript, Chart.js  
**Deep Learning**: TensorFlow / Keras (`newModel.h5`)  
**Video & Audio Processing**: OpenCV, SoundDevice, SoundFile  
**Database**: (Optional - session-based storage used now)  

---

## 🚀 Features

✅ **Video & Audio Recording** during interviews  
✅ **Confidence Score Analysis** using a deep learning model  
✅ **Cheating Detection** using computer vision  
✅ **Interactive UI** (built with Jinja2 templates)  
✅ **Real-time camera streaming**

---

## 🛠 Installation

```bash
git clone https://github.com/codeAshi/mock-interview-fastAPI.git
cd "Mock Interview"
```
▶️ Running the Project
```
uvicorn main:app --reload
Then open: http://127.0.0.1:8000
```
📌 Tech Stack
Backend: FastAPI, Uvicorn

Frontend: HTML (Jinja2 Templates), Chart.js

Deep Learning: TensorFlow / Keras

Media Processing: OpenCV, SoundDevice, SoundFile

🤝 Contributing
Pull requests are welcome! If you find bugs or want new features, feel free to open an issue.

👨‍💻 Author
Ashish Bedare
📧 bedareashish@gmail.com
🌐 https://github.com/codeAshi

# 🎥 Mock Interview System with Confidence Analysis (FastAPI) flow
Mock Interview/

│

├── main.py # FastAPI main app

├── cheating_detector.py # Cheating detection logic

├── newModel.h5 # Deep learning model for confidence analysis

│

├── templates/ # HTML templates (Jinja2)

│              ├── index.html

│              ├── interview.html

│              ├── confidence-score.html

│              └── preview.html

│

├── static/ # CSS & other static files

│              └── style.css

│
├── recordings/ # Saved video recordings

├── audio_recordings/ # Saved audio recordings

