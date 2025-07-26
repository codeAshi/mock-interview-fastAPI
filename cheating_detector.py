import cv2
import os

def detect_cheating_clips(video_list):
    cheating_clips = []

    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cheating_detected = False

        # Example simple logic: if too many frames are dark → assume cheating
        dark_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.mean(gray)[0] < 40:  # very dark
                dark_frames += 1
        cap.release()

        if dark_frames > frame_count * 0.3:  # 30%+ dark frames → cheating
            cheating_clips.append(video_path.replace("\\", "/"))

    return cheating_clips
